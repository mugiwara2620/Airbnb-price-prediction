import pandas as pd

df = pd.DataFrame(
    [
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7],
    ],
    columns=["row1", "row2", "row3", "row4", "row2", "row3", "row4"],
)
last_unnamed_array = df.iloc[:, 0:-1]
last_unnamed_array
