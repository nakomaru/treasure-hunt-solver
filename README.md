# Treasure Hunt Solver

A probability solver for the **Treasure Hunt** minigame from some Blue Archive
events. You flip tiles on a grid to uncover hidden prizes; this tool tells you
which tile to flip next. It runs entirely in your browser.

**▶ [Open the solver](https://nakomaru.github.io/treasure-hunt-solver/)**

## The minigame

A **9 × 5** grid hides a set of prizes, each a rectangular block of tiles.
Flipping a tile reveals empty space or the slice of prize artwork under it.

## How it works

The solver enumerates every arrangement of the remaining prizes consistent
with what you've revealed, then counts, for each unflipped tile, the fraction
of arrangements where a prize covers it.

## Using it

1. **Set the prizes in play.** Each prize card takes a shape (tap to pick one)
   and a count of 0–5. The cards start with a random mix each load.
2. The board lists the hit chance of each remaining tile. Reveal the highest
   one in game.
3. If the tile was **empty**, mark it as **Miss ✕** with left click.
4. If it was a **prize** → click or drag its shape from the prize card onto the
   board.
5. You may also double click or right click to mark it as a **Hit ●** if you
   are unsure of its adjacent tiles (unlikely).

## Running locally

Open `index.html` in a browser.

## License

[CC0](https://creativecommons.org/publicdomain/zero/1.0/)