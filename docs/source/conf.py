# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil
import glob
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LVGP-Bayes'
copyright = '2023, Northwestern University'
author = 'Suraj Yerramilli'
release = '0.3.0'
version = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.githubpages',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# Exclude build directory and Jupyter backup files
exclude_patterns = ['_build', '**.ipynb_checkpoints']

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Intersphinx configuration ------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'gpytorch': ('https://docs.gpytorch.ai/en/stable/', None),
}

# -- Additional configuration -------------------------------------------------
# Add any custom CSS or JavaScript files
# html_css_files = []
# html_js_files = []
# -- Copy notebooks to source directory -------------------------------------
def copy_notebooks_to_source():
    """Copy notebooks from examples/ to docs/source/examples/ for rendering."""
    # Define source and destination directories
    examples_src = os.path.abspath('../../notebooks')
    examples_dst = os.path.abspath('./examples')

    if os.path.exists(examples_dst):
        shutil.rmtree(examples_dst)

    # Create destination directory if it doesn't exist
    os.makedirs(examples_dst, exist_ok=True)

    # Copy all .ipynb files
    if os.path.exists(examples_src):
        notebook_files = []
        for pattern in ('*.ipynb', '*.rst'):
            notebook_files.extend(glob.glob(os.path.join(examples_src, pattern)))
        notebook_files.sort()
        for notebook in notebook_files:
            filename = os.path.basename(notebook)
            dst_path = os.path.join(examples_dst, filename)
            shutil.copy2(notebook, dst_path)
            print(f"Copied {filename} to docs/source/examples/")
    else:
        print(f"Warning: Examples directory {examples_src} not found")

# Copy notebooks before building
copy_notebooks_to_source()
