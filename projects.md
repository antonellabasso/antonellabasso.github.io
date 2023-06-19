---
layout: default
title: "Projects"
---

<!--
{% if site.show_excerpts %}
  {% include home.html %}
{% else %}
  {% include archive.html title="Projects" %}
{% endif %}
--->

{% for staff_member in site.staff_members %}
  <h2>
    <a href="{{ staff_member.url }}">
      {{ staff_member.name }} - {{ staff_member.position }}
    </a>
  </h2>
  <p>{{ staff_member.content | markdownify }}</p>
{% endfor %}
