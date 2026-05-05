import pandas as pd
from pathlib import Path

url = "https://developers.google.com/public-data/docs/canonical/countries_csv"

tables = pd.read_html(url)

countries_df = tables[0]


data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)

countries_df.to_csv(data_dir / "countries_coords.csv", index=False)
print("saved")

