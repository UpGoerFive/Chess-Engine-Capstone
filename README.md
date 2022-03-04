# Chess Engine Capstone

A project to build a chess engine using Convolutional Neural Networks as evaluation metrics.

## Model Selection

Modern chess engines generally fall into two categories. More conventional engines consist of a searching algorithm that makes use of a handmade evaluation metric for the nodes, whereas deep learning engines, such as Alpha Zero or Leela Chess Zero, use reinforcement learning to train. This project is focused on applying neural networking techniques to the evaluation metric of the first type of engine.

Two things make using neural networks for evaluation metrics a viable application. The first is that the evaluation metric needs to return a value for the position based on the current state of the board alone; calculating moves ahead is what the search algorithm is for. The second is that evaluation metrics so far have tended to be manually built and tuned. Using neural networks in this context demonstrates how other evaluation problems could be similarly dealt with. I've focused on convolutional neural nets for this project, in the hope that the network could pay attention to positional features.

## Data

This project uses Lichess puzzles to train and evaluate on. The puzzle data can be downloaded [here](https://database.lichess.org/#puzzles). After unziping into the main repository directory, the `dataorg` notebook can be used to generate the train test split, and advance each FEN by one move (needed because of how Lichess hands off the data). The second half of that notebook produces data for early modeling that is not generally needed otherwise and will take a while to run. I've left it in for completeness and reproducibility of the early models.

The majority of the data preparation is devoted to converting from FEN positions, to an 8x8x13 array. Each channel represents one piece type of one color (with one channel for empty) with pieces for that channel represented as 1 and others as 0. This allows us to pass the data into a network similarly to how image data is often converted to an array first.

Based on how many positions the conversion produced, a single puzzle position has around 32 possible moves, and since these are "Only move" puzzles, only 1 of those moves is labeled as correct. This produces a large imbalance, and is the reason why the early models score around 96% accuracy on the validation data, despite being quite terrible. Guessing "bad move" for every single entry would produce around 97% accuracy.

To correct for this, later models only train on at most 4 bad positions in addition to the correct one (1 in 5 positions in the data are correct moves). This also reduced the the computation needed to convert from FENs for all training moves, making it possible to use the full dataset in training and evaluation, which was not feasible for me before.

## Parts of the Engine

This project uses [Python-Chess](https://github.com/niklasf/python-chess) for all UCI functions. This includes generating legal moves, and getting board FENs. To enable play on Lichess, I've forked the [`lichess-bot`](https://github.com/UpGoerFive/lichess-bot) repository as recommended. The bot can be run on lichess if desired with the instructions provided, but you'll need a bot account on lichess with an Oauth token. Models can be run locally for play using the `rolloutboard` notebook. Available models to choose from are in the Models directory.

Both modeling notebooks can be run, but require different versions of the data generated from `dataorg`.

## Performance

As stated above, early models achieved 96% accuracy, which while worse than a dumb model, does at least indicate there is something the network is picking up on which causes it to guess a position is correct. The model that trained on less imbalanced and more data achieved around 88% accuracy, as compared to an accuracy of around 80% if a model predicted "bad move" for every entry. Both engines do not perform particularly well in actual play, but this is not super surprising on their own without a searching algorithm.

## Play on Lichess

If you'd like to play the most recent version of the bot, you can find it [here](https://lichess.org/@/PuzzledBot) on Lichess.
