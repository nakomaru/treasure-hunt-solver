# Treasure Hunt Solver

A probability solver for the **Treasure Hunt** minigame from some Blue Archive
events. You flip tiles on a grid to uncover hidden prizes; this tool tells you
which tile to flip next. It runs entirely in your browser.

**▶ [Open the solver](https://nakomaru.github.io/treasure-hunt-solver/)**

## The minigame

A **9 × 5** grid hides a set of prizes, each a rectangular block of tiles.
Flipping a tile reveals empty space or the slice of prize artwork under it.

## How it works

The solver counts every arrangement of the remaining prizes consistent with
what you've revealed, then shows, for each unflipped tile, the fraction of
arrangements where a prize covers it. Counting runs as a dynamic program over
the board column by column, so even opening boards with billions of
arrangements solve in milliseconds.

The ★ starts on the highest-chance tile. In the background the solver also
searches for the flip that minimizes expected misses under optimal play; when
the board is small enough for that search to finish, the ★ moves to the
optimal tile and the status line shows the expected misses left.

## Using it

1. **Set the prizes in play.** Each prize card takes a shape (tap to pick one)
   and a count of 0–6. A clean load starts with a random mix and is saved to
   your browser's local storage from then on, so a refresh or revisit keeps
   the same board; **New random board** reseeds it at any time.
2. The board lists the hit chance of each remaining tile. Reveal the ★ tile
   in game.
3. If the tile was **empty**, mark it as **Miss ✕** with left click.
4. If it was a **prize** → click or drag its shape from the prize card onto the
   board.
5. You may also double click or right click to mark it as a **Hit ●** if you
   are unsure of its adjacent tiles (unlikely).

## Running locally

Open `index.html` in a browser.

`node bench.js [games]` checks the engine in `index.html` against a
brute-force counter, times it, and plays paired simulated games to compare
the page's policy with always flipping the highest-chance tile. Results go to
`bench-report.txt`.

## License

[CC0](https://creativecommons.org/publicdomain/zero/1.0/)