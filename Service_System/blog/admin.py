from django.contrib import admin

from blog.models import Post, Session

# Register your models here.
admin.site.register(Post)
admin.site.register(Session)
