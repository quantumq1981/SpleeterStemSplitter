from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('api/search/', views.YouTubeSearchView.as_view()),
    path('api/source-file/all/', views.SourceFileListView.as_view()),
    path(
        'api/source-file/file/',
        views.SourceFileView.as_view({
            'post': 'create',
            'delete': 'perform_destroy'
        })),
    path('api/source-file/youtube/', views.YTLinkInfoView.as_view()),
    path('api/source-track/', views.SourceTrackListView.as_view()),
    path('api/source-track/<uuid:id>/',
         views.SourceTrackRetrieveUpdateDestroyView.as_view()),
    path('api/source-track/file/', views.FileSourceTrackView.as_view()),
    path('api/source-track/youtube/', views.YTSourceTrackView.as_view()),
    path('api/mix/static/', views.StaticMixCreateView.as_view()),
    path('api/mix/static/<uuid:id>/',
         views.StaticMixRetrieveDestroyView.as_view()),
    path('api/mix/dynamic/', views.DynamicMixCreateView.as_view()),
    path('api/mix/dynamic/<uuid:id>/',
         views.DynamicMixRetrieveDestroyView.as_view()),
    path('api/task/', views.YTAudioDownloadTaskListView.as_view()),
    path('api/task/<uuid:id>/',
         views.YTAudioDownloadTaskRetrieveView.as_view()),
    # Chord Analysis endpoints
    path('api/chord-analysis/', views.ChordAnalysisCreateView.as_view()),
    path('api/chord-analysis/<uuid:id>/',
         views.ChordAnalysisRetrieveDestroyView.as_view()),
    path('api/chord-analysis/track/<uuid:track_id>/',
         views.ChordAnalysisByTrackView.as_view()),
    path('api/chord-analysis/<uuid:id>/chart/',
         views.ChordAnalysisChartView.as_view()),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
