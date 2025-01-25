---
title: "Articles by Year"
permalink: /articles/
layout: posts
author_profile: true
---

# Posts for articles

{% for post in site.categories.articles %}
- [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}
