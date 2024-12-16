from django.shortcuts import redirect, render 
from django.utils import timezone

from blog.forms import PostForm 
from .models import Post, Comment, Session
from django.shortcuts import render, get_object_or_404
from rest_framework import viewsets, status
from .serializers import PostSerializer, CommentSerializer, SessionSerializer
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated

class blogImage(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

    def create(self, request, *args, **kwargs):
        # Print various parts of the request for debugging
        print("Headers:", request.headers)
        print("Keys:"   , request.data.keys())
        print("Body:", request.data.get('image'))
        # Call the parent create method to process the request
        serializer = self.get_serializer(data=request.data)
        # serializer.is_valid(raise_exception=True)
        if not serializer.is_valid():
            print("Validation errors:", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        print(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def toggle_favorite(self, request, pk=None):
        post = self.get_object()
        user = request.user
        if user.is_authenticated:
            if user in post.favorites.all():
                post.favorites.remove(user)
                return Response({'status': 'removed from favorites'}, status=status.HTTP_200_OK)
            else:
                post.favorites.add(user)
                return Response({'status': 'added to favorites'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

class SessionViewSet(viewsets.ModelViewSet):
    queryset = Session.objects.all()
    serializer_class = SessionSerializer

    @action(detail=True, methods=['get'])
    def session_statistics(self, request, pk=None):
        session = self.get_object()
        
        # Get all images/posts for this session
        posts = Post.objects.filter(session_id=session.session_id)
        
        # Count occurrences of each state
        state_counts = {
            'LONG_ABSENCE': 0,
            'EYES_CLOSED_LONG': 0,
            'NORMAL': 0
        }
        
        for post in posts:
            if 'LONG_ABSENCE' in post.text:
                state_counts['LONG_ABSENCE'] += 1
            elif 'EYES_CLOSED_LONG' in post.text:
                state_counts['EYES_CLOSED_LONG'] += 1
            elif 'NORMAL' in post.text:
                state_counts['NORMAL'] += 1
        print(state_counts)
        for post in posts:
            print(post.text)

        # Get list of image IDs
        images = posts.values('id', 'title', 'published_date', 'text', 'image')
        
        return Response({
            'session_id': session.session_id,
            'session_start': session.session_start_date,
            'session_end': session.session_end_date,
            'state_statistics': {
                'long_absence_count': state_counts['LONG_ABSENCE'],
                'eyes_closed_count': state_counts['EYES_CLOSED_LONG'],
                'normal_count': state_counts['NORMAL']
            },
            'images': list(images)
        })

def post_list(request):
    posts = Post.objects.filter().order_by('published_date') 
    return render(request, 'blog/post_list.html', {'posts': posts})
def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'blog/post_detail.html', {'post': post})

def post_new(request):
    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def post_edit(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.method == "POST":
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = timezone.now()
            post.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = PostForm(instance=post)
        return render(request, 'blog/post_edit.html', {'form': form})