from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QFont, QColor
from PyQt5.QtCore import QRegExp

class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code."""
    
    def __init__(self, document):
        super().__init__(document)
        self.rules = []
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#c586c0"))  # Purple
        keyword_format.setFontWeight(QFont.Bold)
        
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "exec", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda",
            "not", "or", "pass", "print", "raise", "return", "try",
            "while", "with", "yield", "None", "True", "False"
        ]
        
        for word in keywords:
            pattern = QRegExp(r"\b" + word + r"\b")
            self.rules.append((pattern, keyword_format))
            
        # Class names
        class_format = QTextCharFormat()
        class_format.setForeground(QColor("#4EC9B0"))  # Teal
        class_format.setFontWeight(QFont.Bold)
        self.rules.append((QRegExp(r"\bclass\s*(\w+)"), class_format))
        
        # Function names
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))  # Light Yellow
        self.rules.append((QRegExp(r"\bdef\s*(\w+)"), function_format))
        
        # Decorators
        decorator_format = QTextCharFormat()
        decorator_format.setForeground(QColor("#DCDCAA"))  # Yellow
        self.rules.append((QRegExp(r"@[^\n]+"), decorator_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#b5cea8"))  # Light Green
        self.rules.append((QRegExp(r"\b[+-]?[0-9]+[lL]?\b"), number_format))
        self.rules.append((QRegExp(r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b"), number_format))
        self.rules.append((QRegExp(r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b"), number_format))
        
        # Strings (single and double quotes)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#ce9178"))  # Orange/Brown
        self.rules.append((QRegExp(r"\".*\""), string_format))
        self.rules.append((QRegExp(r"\'.*\'"), string_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))  # Green
        self.rules.append((QRegExp(r"#[^\n]*"), comment_format))
        
    def highlightBlock(self, text):
        for pattern, format in self.rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)
        
        # Multi-line strings (""" or ''')
        # This is a basic implementation and might not cover all edge cases
        self.setCurrentBlockState(0)
