# Generated manually for ChordAnalysis model

import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0013_alter_staticmix_unique_together_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChordAnalysis',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('celery_id', models.UUIDField(blank=True, default=None, null=True)),
                ('key', models.CharField(blank=True, max_length=20)),
                ('key_confidence', models.FloatField(default=0.0)),
                ('result_json', models.JSONField(blank=True, default=dict)),
                ('chart_markdown', models.TextField(blank=True)),
                ('chart_csv', models.TextField(blank=True)),
                ('segment_duration', models.FloatField(default=0.5)),
                ('smoothing', models.FloatField(default=0.6)),
                ('status', models.IntegerField(choices=[(0, 'Queued'), (1, 'In Progress'), (2, 'Done'), (3, 'Error')], default=0)),
                ('error', models.TextField(blank=True)),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('date_finished', models.DateTimeField(default=None, null=True)),
                ('source_track', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chord_analyses', to='api.sourcetrack')),
                ('dynamic_mix', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='chord_analyses', to='api.dynamicmix')),
            ],
            options={
                'verbose_name_plural': 'Chord analyses',
            },
        ),
    ]
