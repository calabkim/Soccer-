# Soccer Match Outcome Prediction

This project predicts the outcomes of soccer matches and the "win-by" margin (goal difference) using historical match and player data. The models built in this project will be used to predict:

- Whether the home team wins or loses (binary classification).
- The goal difference (win-by) between the teams (regression).

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict the results of soccer matches based on past games and player statistics. The main tasks are to:

1. Predict if the home team wins or loses.
2. Predict the goal difference (win-by margin) for the game.

The evaluation is based on:

- **AUC (Area Under the Curve)** for the classification of match outcomes.
- **RMSE (Root Mean Squared Error)** for predicting the goal difference.

## Data Description

The project uses three CSV files as input:

- **games.csv**: Contains match-level data for soccer matches played from 2008 to 2014.
- **players.csv**: Contains player-level statistics such as rating, potential, and work rate.
- **test.csv**: Contains match data for games played in 2015 and 2016, where the goal is to predict outcomes.

### Features:

- **Home/Away Team IDs**: Unique identifiers for home and away teams.
- **Player IDs**: Identifiers for the players in the match.
- **Player Stats**: Aggregated statistics such as overall rating and work rate for players in both teams.

## Feature Engineering

We aggregate player-level statistics for each match by combining the `players.csv` and `games.csv` data:

- **Average player rating**: Aggregated player ratings for both the home and away teams.
- **Potential and work rate**: Other player-level attributes are aggregated similarly.
- **Historical performance**: Historical averages for goals scored and conceded by teams (optional).

These features are then used in machine learning models for classification and regression.

## Modeling Approach

Two models are trained:

1. **Classification**: To predict whether the home team wins or loses.
2. **Regression**: To predict the win-by margin (goal difference).

Both models are trained using **XGBoost**, a powerful machine learning algorithm that is highly efficient for both classification and regression tasks.

### Model 1: Classification

- **Task**: Predict whether the home team wins or loses.
- **Model**: XGBoost with `binary:logistic` objective.

### Model 2: Regression

- **Task**: Predict the goal difference between teams.
- **Model**: XGBoost with `reg:squarederror` objective.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/soccer-outcome-prediction.git
cd soccer-outcome-prediction

