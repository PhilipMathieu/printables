# House rules

Parametric models for FDM printing. The README is the tour; this is how the work
is done. Read it before adding a part or changing a load-bearing one.

## The shape of a part

One module per model in `parts/`, one drawing tool per model in `tools/`, one
test module per model in `tests/`. Shared vocabulary — grids, materials, the
measured machine — lives in `geom/`.

A part module is a frozen `Params` dataclass, a `validate()` that raises with a
sentence explaining what the number breaks, and a `build()` that returns a Part.
Derived quantities are properties on `Params`, named for what they mean, so the
arithmetic has one home and the tests can reach it.

Everything is in millimetres. Parts are modelled **as they print**, which is
usually upside down from how they are used: a shank is always narrower than what
it grows out of, so putting it last makes every change of section a step inward
and no facet overhangs. Say so in the docstring, because it catches people —
including whoever wrote it.

Tests assert design *intent*, measured off the built solid. Never a geometry
hash, never a magic number copied out of a previous run. "The seat is coplanar
with the fence face" is a test; "the volume is 3412.7" is a tripwire that goes
off on every legitimate change and says nothing when a real one breaks.

## The diagram and statics loop

**Anything that carries load gets drawn in section and solved before it is
printed.** Not sketched by hand, not described in prose — drawn from the model,
by code in `tools/`, and then have its equilibrium written down.

This rule exists because prose is where mechanical errors hide. Two got through
here, and both were invisible in a paragraph and obvious in a drawing:

- **The clamp's nut trap opened on the wrong face.** The bolt tip pushes the
  work, so the work pushes back along the bolt, so the bolt drags the nut
  *rearward* — and the pocket was cut into the rear face, the intuitive place,
  since that is the end the bolt goes in. The nut would have walked out on the
  first turn of the screw.
- **The break arm handed its moment to the deck.** Its lever was coplanar with
  the head's seat, so in use it lay flat on the plate — and a lever lying on the
  surface its own fulcrum is in transfers the moment to that surface. On the
  real deck the root would have seen nine percent of the intended load and the
  arm would have read four times too strong, with nothing in the result to say
  so.

Neither was a hard calculation. Both were a free-body diagram that nobody drew.

### The loop

1. **Name the load path in words**, in the module docstring: what pushes what,
   in what order, ending at something that does not move. If you cannot write
   the chain down, you do not have one.
2. **Draw it in section**, with a generator in `tools/` that reads the real
   `Params`. A drawing built from live parameters cannot drift from the STL; a
   drawing made by hand starts drifting the first time a number changes.
3. **Do the statics on the drawing.** Free body, every contact marked, sum of
   forces and sum of moments — then solve for the number the design claims. Do
   this on the drawing rather than in your head, because the drawing is what
   shows you the contacts.
4. **Assert what the statics assumed.** At minimum a test for the contact set:
   the part bears *here* and nowhere else. That is the assumption that fails
   silently, and it is measurable off the solid.

### What the two failures teach, generalised

- **A reaction lands where the part actually touches, not where you meant it
  to.** List every face that can come into contact, then ask which one the load
  rotates the part onto. Both failures above are this mistake.
- **Ask which direction the load pushes the thing that resists it.** A pocket, a
  trap, or a shoulder that opens the way the load pushes is a pocket the load
  empties.
- **A lever that shares a surface with its own fulcrum is not a lever.** The
  surface takes the moment.
- **Take moments about the contact, not the axis you find convenient.** The
  convenient one is usually the one that hides the reaction you forgot.
- **State the assumption in the units the bench will measure in.** A number that
  reads straight off a luggage scale gets checked; one that needs converting
  does not.

### When it applies

Every module in `parts/` is classified in `tests/test_conventions.py`, and a new
one fails the suite until it is. Either it carries load — and then it needs a
drawing generator with a no-argument `draw()`, a committed image in `docs/`, and
the loop above — or it is declared decorative with a one-line reason. There is no
third option and no unclassified default, because "I didn't think it carried
anything" is exactly the state both failures above were written in.

Decorative does not mean careless. It means nothing bears on it, so a drawing is
for showing what it looks like rather than for finding out whether it works.

## Prints and profiles

3mf export needs Bambu Studio's profiles on disk. On a machine without them the
STL is still written and the missing slicer is a note rather than a failure —
keep it that way, and do not let a test depend on the profiles being there.
