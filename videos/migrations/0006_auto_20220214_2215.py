# Generated by Django 2.2.4 on 2022-02-14 14:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0005_auto_20220214_2131'),
    ]

    operations = [
        migrations.AlterField(
            model_name='videos_post',
            name='detect_videos',
            field=models.FileField(blank=True, default='', upload_to='in_out_videos/result'),
        ),
        migrations.AlterField(
            model_name='videos_post',
            name='videos',
            field=models.FileField(blank=True, default='', upload_to='in_out_videos/manipulate'),
        ),
    ]
