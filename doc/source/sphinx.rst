Documentation Guide
===================

This page is mostly intended for my own reference,
as I found myself repeatedly looking up guides of a narrow subset of *reStructuredText*.

Markup Constructs
-----------------

Structured content can be formatted with a **directive**. For example,

+ Comment

  .. code-block:: rst

     .. This is a comment (not rendered in the output).

+ Headings (on way to do it)

  .. code-block:: rst

     Heading 1
     =========

     Heading 2
     ---------

     Heading 3
     *********

+ Unordered lists

  .. code-block:: rst
     
     + Item 1
     + Item 2
     + Item 3

  or just like *Markdown*,

  .. code-block:: rst
     
     - Item 1
     - Item 2
     - Item 3

+ Message boxes (*warning*)

  .. code-block:: rst

     .. warning::

        Warning: This is a warning message.

  which gets rendered into

  .. warning::

     Warning: This is a warning message.

+ Message boxes (*note*)

  .. code-block:: rst

     .. note::

        Note: This is an important message.

  which gets rendered into

  .. note::

     Note: This is an important message.



Build the Documentation
-----------------------

Use the ``Makefile`` to build:

.. code-block:: bash

   # Build the web version
   $ make html

   # Build the pdf version
   $ make pdflatex

   # Report documentation coverage
   $ make coverage


More Information
----------------

  - A demo https://sphinx-rtd-theme.readthedocs.io/en/latest/demo/demo.html
