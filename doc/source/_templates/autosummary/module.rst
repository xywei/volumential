{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

{% block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{# ``qbfem`` is a 2019 finite-element experiment that nothing in the tree
   imports and that the README calls out as unsupported, so it stays out of
   the API reference. #}
{% for item in modules if item.split(".")[-1] != "qbfem" %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
