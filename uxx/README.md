# `uxx`

(Name subject to change)

## What

- Error checked scalar casting
- Non negative signed integers (basically u31 that's just an always positive i32)

## Operations

- Full wrap-around arithmetic
- Debug assertions on overflow

## Missing

- Literals, no support for custom types. `u31(0)` as a temporary workaround.
