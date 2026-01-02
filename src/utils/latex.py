
import tempfile
import subprocess
import shutil
from pathlib import Path

def preview_latex(latex_str, packages, output_path='table_preview.pdf'):
    """
    Compile LaTeX table to PDF
    
    Args:
        latex_str: LaTeX string from style_multiindex_latex()
        output_path: Where to save the final PDF
    """
    output_path = Path(output_path)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Full LaTeX document
        doc = '\\documentclass{article}\n'

        for package in packages:
            doc += '\\usepackage{' + package + '}\n'

        doc += '\\begin{document}\n'
        doc += (latex_str.split('\\n\\n', 1)[1] if '\\n\\n' in latex_str else latex_str)
        #doc += latex_str
        doc += '\\end{document}\n'

        print(doc)

        # Write and compile in temp dir
        tex_file = tmpdir / 'table.tex'
        tex_file.write_text(doc)
        
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'table.tex'],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception("LaTeX compilation error: " + result.stdout + result.stderr)

        # Copy PDF to output location
        pdf_file = tmpdir / 'table.pdf'
        assert pdf_file.exists()

        shutil.copy(pdf_file, output_path)



