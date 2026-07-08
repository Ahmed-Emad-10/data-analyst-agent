import pandas as pd
import traceback
from io import StringIO
from crewai.tools import tool

# The DataFrame is loaded once and reused across all tool calls.
_df: pd.DataFrame | None = None


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load (or return cached) the CSV as a DataFrame."""
    global _df
    if _df is None:
        _df = pd.read_csv(csv_path)
    return _df


def get_schema_context(csv_path: str, sample_rows: int = 5) -> str:
    """
    Return a compact schema description + sample rows.
    This is what the LLM sees instead of the full CSV.
    """
    df = load_dataframe(csv_path)

    lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "",
        "Column dtypes:",
    ]
    for col, dtype in df.dtypes.items():
        null_count = df[col].isna().sum()
        sample_vals = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col!r} ({dtype}) | nulls={null_count} | e.g. {sample_vals}")

    lines += ["", f"First {sample_rows} rows (raw):"]
    lines.append(df.head(sample_rows).to_string(index=False))

    return "\n".join(lines)


@tool("run_pandas_code")
def run_pandas_code(code: str) -> str:
    """
    Execute a Python/pandas code snippet against the customer CSV and return
    the result as a string.

    The DataFrame is already loaded and available as the variable `df`.
    Your code MUST assign the final answer to a variable called `result`.

    Example:
        result = df[df['age'] > 40]['job'].value_counts().head(5).to_string()
    """
    from config import CSV_PATH  # local import to avoid circular

    df = load_dataframe(CSV_PATH)  # noqa: F841  (used inside exec)

    local_vars: dict = {"df": df, "pd": pd}

    try:
        exec(compile(code, "<agent_code>", "exec"), local_vars)  # noqa: S102
    except Exception:
        return f"ERROR during execution:\n{traceback.format_exc()}"

    result = local_vars.get("result")
    if result is None:
        return (
            "ERROR: your code did not assign anything to `result`. "
            "Please assign the final answer to a variable named `result`."
        )

    # Convert common types to readable strings
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False)
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)
