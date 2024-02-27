from pathlib import Path
from pydantic import BaseModel


class Paths(BaseModel):
    # basement
    data_dir: Path = Path("data")
    logs_dir: Path = data_dir / "logs"
    input_dir: Path = data_dir / "input"
    output_dir: Path = data_dir / "output"
    # logs
    logs_filepath: Path = logs_dir / "logs.log"
    # data
    warsaw_weather_filepath: Path = input_dir / "last_3_year_tavg.csv"
