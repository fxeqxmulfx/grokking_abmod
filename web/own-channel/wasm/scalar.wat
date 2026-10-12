;; Scalar equivalents of the SIMD vector kernels. All lengths are multiples
;; of four; the shared training module supplies bounded pointers and lengths.
(func $clear (param $dst i32) (param $count i32)
  (local $end i32)
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (f32.store (local.get $dst) (f32.const 0))
    (local.set $dst (i32.add (local.get $dst) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $axpy (param $dst i32) (param $src i32) (param $count i32) (param $scale f32)
  (local $end i32)
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (f32.store (local.get $dst) (f32.add (f32.load (local.get $dst)) (f32.mul (f32.load (local.get $src)) (local.get $scale))))
    (local.set $src (i32.add (local.get $src) (i32.const 4)))
    (local.set $dst (i32.add (local.get $dst) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $dot (param $a i32) (param $b i32) (param $count i32) (result f32)
  (local $end i32) (local $sum f32)
  (local.set $end (i32.add (local.get $a) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (local.set $sum (f32.add (local.get $sum) (f32.mul (f32.load (local.get $a)) (f32.load (local.get $b)))))
    (local.set $b (i32.add (local.get $b) (i32.const 4)))
    (local.set $a (i32.add (local.get $a) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $a) (local.get $end))))
  (local.get $sum))

(func $multiply (param $dst i32) (param $count i32) (param $scale f32)
  (local $end i32)
  (local.set $end (i32.add (local.get $dst) (i32.mul (local.get $count) (i32.const 4))))
  (loop $loop
    (f32.store (local.get $dst) (f32.mul (f32.load (local.get $dst)) (local.get $scale)))
    (local.set $dst (i32.add (local.get $dst) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $dst) (local.get $end)))))

(func $input_kernel (param $a i32) (param $b i32)
  (local $at i32) (local $h f32)
  (loop $loop
    (local.set $h (f32.max (f32.const 0) (f32.add (f32.load (i32.add (local.get $a) (local.get $at))) (f32.load (i32.add (local.get $b) (local.get $at))))))
    (f32.store (i32.add (global.get $input) (local.get $at)) (local.get $h))
    (f32.store (i32.add (global.get $hidden) (local.get $at)) (f32.mul (local.get $h) (f32.load (i32.add (global.get $products) (local.get $at)))))
    (local.set $at (i32.add (local.get $at) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $at) (i32.const 256)))))

(func $exponentials (param $maximum f32)
  (local $at i32)
  (loop $loop
    (f32.store (i32.add (global.get $errors) (local.get $at))
      (call $exp (f32.sub (f32.load (i32.add (global.get $logits) (local.get $at))) (local.get $maximum))))
    (local.set $at (i32.add (local.get $at) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $at) (i32.const 388))))
  (f32.store offset=388 (global.get $errors) (f32.const 0))
  (f32.store offset=392 (global.get $errors) (f32.const 0))
  (f32.store offset=396 (global.get $errors) (f32.const 0)))

(func $adam (param $lr f32) (param $decay f32) (param $correction1 f32) (param $correction2 f32)
  (local $at i32) (local $g f32) (local $m f32) (local $v f32) (local $w f32)
  (loop $loop
    (local.set $g (f32.load (i32.add (global.get $gradient) (local.get $at))))
    (local.set $w (f32.load (local.get $at)))
    (local.set $m (f32.add (f32.mul (f32.const 0.9) (f32.load (i32.add (global.get $moment1) (local.get $at))))
      (f32.mul (f32.sub (f32.const 1) (f32.const 0.9)) (local.get $g))))
    (local.set $v (f32.add (f32.mul (f32.const 0.98) (f32.load (i32.add (global.get $moment2) (local.get $at))))
      (f32.mul (f32.mul (f32.sub (f32.const 1) (f32.const 0.98)) (local.get $g)) (local.get $g))))
    (f32.store (i32.add (global.get $moment1) (local.get $at)) (local.get $m))
    (f32.store (i32.add (global.get $moment2) (local.get $at)) (local.get $v))
    (f32.store (local.get $at) (f32.sub (local.get $w) (f32.mul (local.get $lr)
      (f32.add (f32.div (f32.div (local.get $m) (local.get $correction1))
        (f32.add (f32.sqrt (f32.div (local.get $v) (local.get $correction2))) (f32.const 1e-8))) (f32.mul (local.get $decay) (local.get $w))))))
    (local.set $at (i32.add (local.get $at) (i32.const 4))) (br_if $loop (i32.lt_u (local.get $at) (i32.const 75520)))))
