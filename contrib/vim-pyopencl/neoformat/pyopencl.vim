" Neoformat Formatter for PyOpenCL
" -----------------------------
"
" (C) Xiaoyu Wei 2017, MIT license
"
" Installation:
" Just drop this file into autoload/neoformat/formatters/pyopencl.vim
" Make sure you have yapf installed and added to PATH.
"
" More about Neoformat: https://github.com/sbdchd/neoformat
"
" You may also include a line
" vim: filetype=pyopencl.python
" at the end of your file to set the file type automatically.
"

function! neoformat#formatters#pyopencl#enabled() abort
    return ['yapf']
endfunction

function! neoformat#formatters#pyopencl#yapf() abort
    return {
                \ 'exe': 'yapf',
                \ 'args': ["--style='{based_on_style: pep8, column_limit: 80}'"],
                \ 'stdin': 1
                \ }
endfunction
