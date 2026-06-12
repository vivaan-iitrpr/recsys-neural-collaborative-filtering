# Dataset

This project uses the **MovieLens 25M** dataset from GroupLens Research.

## Download Instructions

1. Visit: https://grouplens.org/datasets/movielens/25m/
2. Download `ml-25m.zip` (~250MB)
3. Extract to this directory so the structure looks like:

```
data/
├── README.md
└── ml-25m/
    ├── ratings.csv      (25M ratings)
    ├── movies.csv       (62,000 movies)
    ├── tags.csv
    ├── links.csv
    └── genome-scores.csv
```

## Dataset Stats
- **25,000,095** ratings
- **162,541** users
- **62,423** movies
- Rating scale: 0.5 to 5.0
- Sparsity: ~99.7%

## Citation
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
