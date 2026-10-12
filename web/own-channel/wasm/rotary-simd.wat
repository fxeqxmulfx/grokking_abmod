;; Exact pairwise quarter-turn addition. The transpose reverses both signs.
(func $rotary_add (param $dst i32) (param $src i32) (param $transpose i32)
  (local $at i32) (local $value v128) (local $sign v128)
  (local.set $sign (v128.const i32x4 -2147483648 0 -2147483648 0))
  (if (local.get $transpose) (then
    (local.set $sign (v128.const i32x4 0 -2147483648 0 -2147483648))))
  (loop $pairs
    (local.set $value (v128.load (i32.add (local.get $src) (local.get $at))))
    (local.set $value (v128.xor (i8x16.shuffle 4 5 6 7 0 1 2 3 12 13 14 15 8 9 10 11
      (local.get $value) (local.get $value)) (local.get $sign)))
    (v128.store (i32.add (local.get $dst) (local.get $at))
      (f32x4.add (v128.load (i32.add (local.get $dst) (local.get $at))) (local.get $value)))
    (local.set $at (i32.add (local.get $at) (i32.const 16)))
    (br_if $pairs (i32.lt_u (local.get $at) (i32.const 512)))))
