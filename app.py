#!/usr/bin/env python

import sys

import flet as ft
import torch
import whisper

from constants import (
    MSG_READY,
    MSG_LOADING_MODEL,
    MSG_TRANSCRIBING
) 
import customprogressbar
from common import transcribe_listeners

transcribe_module = sys.modules['whisper.transcribe']
transcribe_module.tqdm.tqdm = customprogressbar.CustomProgressBar

def main(page: ft.Page):
    def update_status(message, progress_value):
        status_text.value = message
        progress_bar.value = progress_value
        page.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        audio_file_path.value = ''
        if e.files is not None:
            audio_file_path.value = e.files[0].path
        page.update()

    def button_clicked(e):
        update_status(MSG_LOADING_MODEL, None)
        cuda_available = torch.cuda.is_available()
        model = whisper.load_model(model_type.value,
                                   device='cuda' if cuda_available else 'cpu')
        update_status(MSG_TRANSCRIBING, 0)
        result = model.transcribe(audio_file_path.value)
        new_value = ''
        for r in result['segments']:
            new_value += f'{r["text"]}\n'
        transcribed_text.value = new_value
        page.update()
        update_status(MSG_READY, 0)

    status_text = ft.Text(MSG_READY)
    progress_bar = ft.ProgressBar()
    progress_bar.value = 0
    model_type = ft.Dropdown(
        options=[
            ft.dropdown.Option('tiny'),
            ft.dropdown.Option('base'),
            ft.dropdown.Option('small'),
            ft.dropdown.Option('medium'),
            ft.dropdown.Option('large')
        ]
    )
    model_type.value = 'small'
    audio_file_path = ft.Text()
    transcribed_text = ft.TextField(max_lines=20)
    transcribe_listeners.append({'page': page, 'progress_bar': progress_bar})

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(pick_files_dialog)

    page.add(
        status_text,
        progress_bar,
        ft.Row(
            [
                ft.Text('Whisper model'),
                model_type
            ]
        ),
        ft.Row(
            [
                ft.ElevatedButton(
                    "Pick a file",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False,
                        file_type=ft.FilePickerFileType.AUDIO,
                    ),
                ),
                audio_file_path,
            ]
        ),
        ft.ElevatedButton(
            'Transcribe',
            on_click=button_clicked,
            data=0),
        transcribed_text
    )

ft.app(target=main)
