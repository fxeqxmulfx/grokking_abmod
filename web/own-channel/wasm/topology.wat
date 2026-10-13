;; Software line rasterizer. Accumulate horizontal spans in a difference
;; buffer, then prefix-sum each row. Every connection contributes once per row.
(module
  (memory (export "memory") 1 8192)
  ;; @RASTER_VECTOR
  (func $line (param $ax f32) (param $ay f32) (param $bx f32) (param $by f32)
    (param $color i32) (param $scratch i32) (param $width i32) (param $height i32)
    (local $y i32) (local $end i32) (local $left i32) (local $right i32)
    (local $x f32) (local $step f32) (local $half f32) (local $padding f32) (local $inside i32) (local $row i32)
    (if (f32.le (local.get $by) (local.get $ay)) (then (return)))
    (local.set $y (i32.trunc_f32_s (f32.max (f32.const 0) (f32.ceil (local.get $ay)))))
    (local.set $end (i32.trunc_f32_s (f32.min (f32.convert_i32_s (local.get $height)) (f32.ceil (local.get $by)))))
    (if (i32.ge_s (local.get $y) (local.get $end)) (then (return)))
    (local.set $step (f32.div (f32.sub (local.get $bx) (local.get $ax)) (f32.sub (local.get $by) (local.get $ay))))
    (local.set $half (f32.add (f32.mul (f32.abs (local.get $step)) (f32.const 0.5)) (f32.const 0.5)))
    ;; Most fitted connections are entirely inside the viewport. Their spans
    ;; need no per-row clamping; the padding covers fractional endpoint rows.
    (local.set $padding (f32.add (f32.abs (local.get $step)) (f32.const 1)))
    (local.set $inside (i32.and
      (i32.and (f32.ge (local.get $ax) (local.get $padding)) (f32.ge (local.get $bx) (local.get $padding)))
      (i32.and (f32.le (local.get $ax) (f32.sub (f32.convert_i32_s (local.get $width)) (local.get $padding)))
        (f32.le (local.get $bx) (f32.sub (f32.convert_i32_s (local.get $width)) (local.get $padding))))))
    (local.set $x (f32.add (local.get $ax) (f32.mul (local.get $step) (f32.sub (f32.add (f32.convert_i32_s (local.get $y)) (f32.const 0.5)) (local.get $ay)))))
    (local.set $row (i32.add (local.get $scratch) (i32.mul (i32.mul (local.get $y) (i32.add (local.get $width) (i32.const 1))) (i32.const 16))))
    (loop $rows
      (if (local.get $inside) (then
        (local.set $left (i32.trunc_f32_s (f32.ceil (f32.sub (local.get $x) (local.get $half)))))
        (local.set $right (i32.trunc_f32_s (f32.floor (f32.add (local.get $x) (local.get $half))))))
      (else
        (local.set $left (i32.trunc_f32_s (f32.max (f32.const 0) (f32.ceil (f32.sub (local.get $x) (local.get $half))))))
        (local.set $right (i32.trunc_f32_s (f32.min (f32.convert_i32_s (i32.sub (local.get $width) (i32.const 1))) (f32.floor (f32.add (local.get $x) (local.get $half))))))))
      (if (i32.le_s (local.get $left) (local.get $right)) (then
        (call $add (i32.add (local.get $row) (i32.mul (local.get $left) (i32.const 16))) (local.get $color))
        (call $subtract (i32.add (local.get $row) (i32.mul (i32.add (local.get $right) (i32.const 1)) (i32.const 16))) (local.get $color))))
      (local.set $x (f32.add (local.get $x) (local.get $step)))
      (local.set $row (i32.add (local.get $row) (i32.mul (i32.add (local.get $width) (i32.const 1)) (i32.const 16))))
      (local.set $y (i32.add (local.get $y) (i32.const 1)))
      (br_if $rows (i32.lt_s (local.get $y) (local.get $end)))))
  (func (export "render") (param $codes i32) (param $palette i32) (param $columns i32) (param $residues i32) (param $answers i32)
    (param $scratch i32) (param $pixels i32) (param $width i32) (param $height i32) (param $channels i32)
    (param $y0 f32) (param $y1 f32) (param $y2 f32) (param $y3 f32)
    (local $r i32) (local $c i32) (local $source i32) (local $index i32) (local $code i32) (local $e i32)
    (local $x f32) (local $a f32) (local $b f32) (local $q f32)
    (memory.fill (local.get $scratch) (i32.const 0) (i32.mul (i32.mul (i32.add (local.get $width) (i32.const 1)) (local.get $height)) (i32.const 16)))
    (local.set $e (i32.mul (i32.const 97) (local.get $channels)))
    (loop $residue
      (local.set $a (f32.load (i32.add (local.get $residues) (i32.mul (local.get $r) (i32.const 4)))))
      (local.set $b (f32.load (i32.add (local.get $residues) (i32.mul (i32.add (local.get $r) (i32.const 97)) (i32.const 4)))))
      (local.set $q (f32.load (i32.add (local.get $answers) (i32.mul (local.get $r) (i32.const 4)))))
      (local.set $c (i32.const 0))
      (loop $channel
        (local.set $x (f32.load (i32.add (local.get $columns) (i32.mul (local.get $c) (i32.const 4)))))
        (local.set $index (i32.add (i32.mul (local.get $r) (local.get $channels)) (local.get $c)))
        (local.set $code (i32.load16_u (i32.add (local.get $codes) (i32.mul (local.get $index) (i32.const 2)))))
        (if (i32.ne (local.get $code) (i32.const 65535)) (then
          (call $line (local.get $a) (local.get $y0) (local.get $x) (local.get $y1) (i32.add (local.get $palette) (i32.mul (local.get $code) (i32.const 16))) (local.get $scratch) (local.get $width) (local.get $height))
          (call $line (local.get $x) (local.get $y2) (local.get $q) (local.get $y3) (i32.add (local.get $palette) (i32.mul (local.get $code) (i32.const 16))) (local.get $scratch) (local.get $width) (local.get $height))))
        (local.set $code (i32.load16_u (i32.add (local.get $codes) (i32.mul (i32.xor (local.get $index) (i32.const 1)) (i32.const 2)))))
        (if (i32.ne (local.get $code) (i32.const 65535)) (then
          (if (i32.and (i32.eqz (i32.and (local.get $c) (i32.const 1))) (i32.lt_u (local.get $code) (i32.const 4096))) (then (local.set $code (i32.xor (local.get $code) (i32.const 1024)))))
          (call $line (local.get $b) (local.get $y0) (local.get $x) (local.get $y1) (i32.add (local.get $palette) (i32.mul (local.get $code) (i32.const 16))) (local.get $scratch) (local.get $width) (local.get $height))))
        (local.set $c (i32.add (local.get $c) (i32.const 1))) (br_if $channel (i32.lt_u (local.get $c) (local.get $channels))))
      (local.set $r (i32.add (local.get $r) (i32.const 1))) (br_if $residue (i32.lt_u (local.get $r) (i32.const 97))))
    (loop $source
      (local.set $a (f32.load (i32.add (local.get $columns) (i32.mul (local.get $source) (i32.const 4)))))
      (local.set $c (i32.const 0))
      (loop $output
        (local.set $index (i32.add (local.get $e) (i32.add (i32.mul (local.get $source) (local.get $channels)) (local.get $c))))
        (local.set $code (i32.load16_u (i32.add (local.get $codes) (i32.mul (local.get $index) (i32.const 2)))))
        (if (i32.ne (local.get $code) (i32.const 65535)) (then
          (local.set $b (f32.load (i32.add (local.get $columns) (i32.mul (local.get $c) (i32.const 4)))))
          (call $line (local.get $a) (local.get $y1) (local.get $b) (local.get $y2) (i32.add (local.get $palette) (i32.mul (local.get $code) (i32.const 16))) (local.get $scratch) (local.get $width) (local.get $height))))
        (local.set $c (i32.add (local.get $c) (i32.const 1))) (br_if $output (i32.lt_u (local.get $c) (local.get $channels))))
      (local.set $source (i32.add (local.get $source) (i32.const 1))) (br_if $source (i32.lt_u (local.get $source) (local.get $channels))))
    (call $pixels (local.get $scratch) (local.get $pixels) (local.get $width) (local.get $height)))
  (func $byte (param $value f32) (result i32)
    (i32.trunc_f32_u (f32.min (f32.const 255) (f32.max (f32.const 0) (local.get $value)))))
  (func $pixels (param $scratch i32) (param $pixels i32) (param $width i32) (param $height i32)
    (local $row i32) (local $x i32) (local $source i32) (local $destination i32)
    (local $r f32) (local $g f32) (local $b f32) (local $alpha f32) (local $color i32)
    (local.set $source (local.get $scratch)) (local.set $destination (local.get $pixels))
    (loop $rows
      (local.set $x (i32.const 0)) (local.set $r (f32.const 0)) (local.set $g (f32.const 0)) (local.set $b (f32.const 0)) (local.set $alpha (f32.const 0))
      (loop $columns
        (local.set $r (f32.add (local.get $r) (f32.load (local.get $source))))
        (local.set $g (f32.add (local.get $g) (f32.load offset=4 (local.get $source))))
        (local.set $b (f32.add (local.get $b) (f32.load offset=8 (local.get $source))))
        (local.set $alpha (f32.add (local.get $alpha) (f32.load offset=12 (local.get $source))))
        (local.set $color (i32.const 0))
        (if (f32.gt (local.get $alpha) (f32.const 0.000001)) (then
          (local.set $color (i32.or
            (i32.or (call $byte (f32.div (local.get $r) (local.get $alpha))) (i32.shl (call $byte (f32.div (local.get $g) (local.get $alpha))) (i32.const 8)))
            (i32.or (i32.shl (call $byte (f32.div (local.get $b) (local.get $alpha))) (i32.const 16))
              (i32.shl (call $byte (f32.div (f32.mul (f32.const 255) (local.get $alpha)) (f32.add (f32.const 1) (local.get $alpha)))) (i32.const 24)))))))
        (i32.store (local.get $destination) (local.get $color))
        (local.set $source (i32.add (local.get $source) (i32.const 16))) (local.set $destination (i32.add (local.get $destination) (i32.const 4)))
        (local.set $x (i32.add (local.get $x) (i32.const 1))) (br_if $columns (i32.lt_u (local.get $x) (local.get $width))))
      (local.set $source (i32.add (local.get $source) (i32.const 16)))
      (local.set $row (i32.add (local.get $row) (i32.const 1))) (br_if $rows (i32.lt_u (local.get $row) (local.get $height)))))
)
