import nbformat
from nbformat.reader import NotJSONError

def extract_cells(notebook_path):
    """
    Extracts Python code cells and Markdown cells from a Jupyter Notebook.

    Parameters:
        notebook_path (str): The file path to the Jupyter Notebook.

    Returns:
        code_cells (list): A list containing the source code of all code cells.
        markdown_cells (list): A list containing the source text of all markdown cells.
    """
    try:
        nb = nbformat.read(notebook_path, as_version=4)
    except FileNotFoundError:
        print(f"Error: The file '{notebook_path}' was not found.")
        return [], []
    except NotJSONError:
        print(f"Error: The file '{notebook_path}' is not a valid Jupyter Notebook.")
        return [], []

    cells = []

    for cell in nb.cells:
        if cell.cell_type == 'code':
            cells.append(cell.source)
        elif cell.cell_type == 'markdown':
            cells.append(cell.source)

    return cells


if __name__ == "__main__":
    notebook_file = '/home/eduard/dep-tasks/audio-dl/hw2/homework2.ipynb'
    cells = extract_cells(notebook_file)

    print("Extracted Cells:")
    for i, code in enumerate(cells, 1):
        print(f"\Cell {i}:\n{'-'*40}\n{code}")
