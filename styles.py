from PySide6.QtWidgets import QFrame

PRED_LIT = """
border-radius: 5px;
border: 1px solid black;
background-color: green;
"""

PRED_DIM = """
border-radius: 5px;
border: 1px solid black;
"""

PRED_INACTIVE = """
border-radius: 5px;
border: 1px solid black;
background-color: lightgrey;
"""

LABEL_YES = """
font-weight: bold;
color: green;
"""

LABEL_NO = """
font-weight: bold;
color: red;
"""

def make_sep():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setLineWidth(2)
    sep.setStyleSheet("color: grey;")
    return sep