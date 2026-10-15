;; Four f32 lanes per instruction. No relaxed SIMD or fused operations: fixed
;; SIMD128 keeps f32 semantics consistent across CPU architectures.
(func $clear (param $dst i32) (param $count i32)
  (local $end i32)
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (v128.store (local.get $dst) (v128.const i32x4 0 0 0 0))
    (local.set $dst (i32.add (local.get $dst) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $axpy (param $dst i32) (param $src i32) (param $count i32) (param $scale f32)
  (local $end i32) (local $factor v128)
  (local.set $factor (f32x4.splat (local.get $scale)))
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (v128.store (local.get $dst) (f32x4.add (v128.load (local.get $dst)) (f32x4.mul (v128.load (local.get $src)) (local.get $factor))))
    (local.set $src (i32.add (local.get $src) (i32.const 16)))
    (local.set $dst (i32.add (local.get $dst) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $dot (param $a i32) (param $b i32) (param $count i32) (result f32)
  (local $end i32) (local $sum v128)
  (local.set $end (i32.add (local.get $a) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (local.set $sum (f32x4.add (local.get $sum) (f32x4.mul (v128.load (local.get $a)) (v128.load (local.get $b)))))
    (local.set $b (i32.add (local.get $b) (i32.const 16)))
    (local.set $a (i32.add (local.get $a) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $a) (local.get $end))))
  (f32.add (f32.add (f32x4.extract_lane 0 (local.get $sum)) (f32x4.extract_lane 1 (local.get $sum)))
    (f32.add (f32x4.extract_lane 2 (local.get $sum)) (f32x4.extract_lane 3 (local.get $sum)))))

(func $multiply (param $dst i32) (param $count i32) (param $scale f32)
  (local $end i32) (local $factor v128)
  (local.set $factor (f32x4.splat (local.get $scale)))
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (v128.store (local.get $dst) (f32x4.mul (v128.load (local.get $dst)) (local.get $factor)))
    (local.set $dst (i32.add (local.get $dst) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $maximum (param $slot i32) (result f32)
  ;; Largest of the 97 logits in row slot $slot; its first index goes to $winner.
  ;; This is what a scalar scan with strict comparisons returns: unless the
  ;; first logit is NaN (which then stays), NaN never wins, the value is the
  ;; lane maximum and the winner is the first logit equal to it, even for zeros.
  (local $from i32) (local $at i32) (local $lanes v128) (local $value f32) (local $mask i32)
  (local.set $from (i32.add (global.get $logits) (local.get $slot)))
  (local.set $value (f32.load (local.get $from)))
  (if (f32.ne (local.get $value) (local.get $value)) (then (global.set $winner (i32.const 0)) (return (local.get $value))))
  (local.set $lanes (v128.const f32x4 -inf -inf -inf -inf))
  (loop $lanes_loop
    (local.set $lanes (f32x4.pmax (local.get $lanes) (v128.load (i32.add (local.get $from) (local.get $at)))))
    (local.set $at (i32.add (local.get $at) (i32.const 16))) (br_if $lanes_loop (i32.lt_u (local.get $at) (i32.const 384))))
  (local.set $lanes (f32x4.pmax (local.get $lanes) (i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 (local.get $lanes) (local.get $lanes))))
  (local.set $lanes (f32x4.pmax (local.get $lanes) (i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11 (local.get $lanes) (local.get $lanes))))
  ;; Logit 96 wins only if larger: an equal one comes earlier.
  (local.set $value (f32.load offset=384 (local.get $from)))
  (if (f32.gt (local.get $value) (f32x4.extract_lane 0 (local.get $lanes))) (then (global.set $winner (i32.const 96)) (return (local.get $value))))
  (local.set $at (i32.const 0))
  (loop $scan
    (local.set $mask (i32x4.bitmask (f32x4.eq (v128.load (i32.add (local.get $from) (local.get $at))) (local.get $lanes))))
    (if (local.get $mask) (then
      (global.set $winner (i32.add (i32.shr_u (local.get $at) (i32.const 2)) (i32.ctz (local.get $mask))))
      (return (f32.load (i32.add (local.get $from) (i32.shl (global.get $winner) (i32.const 2)))))))
    (local.set $at (i32.add (local.get $at) (i32.const 16))) (br_if $scan (i32.lt_u (local.get $at) (i32.const 384))))
  unreachable)

(func $exponentials (param $maximum f32) (param $slot i32)
  ;; Exponentials of the logits in row slot $slot, written to its errors.
  ;; Constants live in locals because V8 rebuilds splats inside loops. The
  ;; integral n = nearest(x log2 e) lies in [-126, 0]; adding 1.5 * 2^23 + 127
  ;; leaves n + 127 in the low mantissa bits, which shift into 2^n's exponent.
  ;; Lanes below the cutoff (and NaN) are masked to zero, as before.
  (local $at i32) (local $x v128) (local $n v128) (local $r v128) (local $p v128) (local $normal v128) (local $from i32) (local $to i32)
  (local $max v128) (local $cutoff v128) (local $log2e v128) (local $magic v128) (local $high v128) (local $low v128)
  (local $c7 v128) (local $c6 v128) (local $c5 v128) (local $c4 v128) (local $c3 v128) (local $c2 v128) (local $one v128)
  (local.set $from (i32.add (global.get $logits) (local.get $slot)))
  (local.set $to (i32.add (global.get $errors) (local.get $slot)))
  (local.set $max (f32x4.splat (local.get $maximum)))
  (local.set $cutoff (f32x4.splat (f32.const -87.336544)))
  (local.set $log2e (f32x4.splat (f32.const 1.4426950408889634)))
  (local.set $magic (f32x4.splat (f32.const 12583039)))
  (local.set $high (f32x4.splat (f32.const 0.693359375)))
  (local.set $low (f32x4.splat (f32.const -0.00021219444005469058)))
  (local.set $c7 (f32x4.splat (f32.const 0.0001984126984126984)))
  (local.set $c6 (f32x4.splat (f32.const 0.001388888888888889)))
  (local.set $c5 (f32x4.splat (f32.const 0.008333333333333333)))
  (local.set $c4 (f32x4.splat (f32.const 0.041666666666666664)))
  (local.set $c3 (f32x4.splat (f32.const 0.16666666666666666)))
  (local.set $c2 (f32x4.splat (f32.const 0.5)))
  (local.set $one (f32x4.splat (f32.const 1)))
  (loop $loop
    (local.set $x (f32x4.sub (v128.load (i32.add (local.get $from) (local.get $at))) (local.get $max)))
    (local.set $normal (f32x4.ge (local.get $x) (local.get $cutoff)))
    (local.set $x (f32x4.pmax (local.get $x) (local.get $cutoff)))
    (local.set $n (f32x4.nearest (f32x4.mul (local.get $x) (local.get $log2e))))
    (local.set $r (f32x4.sub (f32x4.sub (local.get $x) (f32x4.mul (local.get $n) (local.get $high)))
      (f32x4.mul (local.get $n) (local.get $low))))
    (local.set $p (f32x4.add (local.get $c6) (f32x4.mul (local.get $r) (local.get $c7))))
    (local.set $p (f32x4.add (local.get $c5) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (local.get $c4) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (local.get $c3) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (local.get $c2) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (local.get $one) (f32x4.mul (local.get $r) (local.get $p))))
    (local.set $p (f32x4.add (local.get $one) (f32x4.mul (local.get $r) (local.get $p))))
    (v128.store (i32.add (local.get $to) (local.get $at))
      (v128.and (f32x4.mul (local.get $p) (i32x4.shl (f32x4.add (local.get $n) (local.get $magic)) (i32.const 23))) (local.get $normal)))
    (local.set $at (i32.add (local.get $at) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $at) (i32.const 400))))
  (f32.store offset=388 (local.get $to) (f32.const 0))
  (f32.store offset=392 (local.get $to) (f32.const 0))
  (f32.store offset=396 (local.get $to) (f32.const 0)))

(func $adam (param $lr f32) (param $decay f32) (param $correction1 f32) (param $correction2 f32)
  (local $at i32) (local $g v128) (local $m v128) (local $v v128) (local $w v128)
  (local $rate v128) (local $dec v128) (local $c1 v128) (local $c2 v128)
  (local.set $rate (f32x4.splat (local.get $lr))) (local.set $dec (f32x4.splat (local.get $decay)))
  (local.set $c1 (f32x4.splat (local.get $correction1))) (local.set $c2 (f32x4.splat (local.get $correction2)))
  (loop $loop
    (local.set $g (v128.load (i32.add (global.get $gradient) (local.get $at))))
    (local.set $w (v128.load (local.get $at)))
    (local.set $m (f32x4.add (f32x4.mul (f32x4.splat (f32.const 0.9)) (v128.load (i32.add (global.get $moment1) (local.get $at))))
      (f32x4.mul (f32x4.splat (f32.sub (f32.const 1) (f32.const 0.9))) (local.get $g))))
    (local.set $v (f32x4.add (f32x4.mul (f32x4.splat (f32.const 0.98)) (v128.load (i32.add (global.get $moment2) (local.get $at))))
      (f32x4.mul (f32x4.mul (f32x4.splat (f32.sub (f32.const 1) (f32.const 0.98))) (local.get $g)) (local.get $g))))
    (v128.store (i32.add (global.get $moment1) (local.get $at)) (local.get $m))
    (v128.store (i32.add (global.get $moment2) (local.get $at)) (local.get $v))
    (v128.store (local.get $at) (f32x4.sub (local.get $w) (f32x4.mul (local.get $rate)
      (f32x4.add (f32x4.div (f32x4.div (local.get $m) (local.get $c1))
        (f32x4.add (f32x4.sqrt (f32x4.div (local.get $v) (local.get $c2))) (f32x4.splat (f32.const 1e-8)))) (f32x4.mul (local.get $dec) (local.get $w))))))
    (local.set $at (i32.add (local.get $at) (i32.const 16))) (br_if $loop (i32.lt_u (local.get $at) (i32.const @WEIGHT_BYTES)))))
