# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:59:03 2018

@author: chen
"""

import librosa
import tensorflow as tf
import pyaudio
import sys
import time
import array
import numpy as np
import queue
from collections import deque
from easydict import EasyDict

args=EasyDict()
args.duration=1
args.rate=44100
args.hop_length=80
args.n_mels=40
args.rt_oversamples=10
args.samples=args.rate*args.duration
args.list=(args.n_mels,1+int(np.floor(args.samples/args.hop_length)),1)
args.rt_process_count=1
args.model='alexnet'
args.rt_chunk_samples = args.rate // args.rt_oversamples
args.mels_onestep_samples = args.rt_chunk_samples * args.rt_process_count
args.mels_convert_samples=args.samples + args.mels_onestep_samples
args.fmax = args.rate // 2
args.n_fft = args.n_mels * 20
args.labels = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
               'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
               'drilling']

args.mels_convert_samples = args.samples + args.mels_onestep_samples

graph_file="E:\\ML\\UrbanSound8K\\code\\UrbanSound8K\\others sussfer\\first\\99.pb"

def audio_to_melspectrogram(args, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=args.rate,
                                                 n_mels=args.n_mels,
                                                 hop_length=args.hop_length,
                                                 n_fft=args.n_fft,
                                                 fmin=20,
                                                 fmax=args.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def callback(in_data, frame_count, time_info, status):
    wave = array.array('b', in_data)
    raw_frames.put(wave, True)
    return (None, pyaudio.paContinue)

def on_predicted(ensembled_pred):
    result = np.argmax(ensembled_pred)
    print(args.labels[result], ensembled_pred[result])
def samplewise_normalize_audio_X(X):
    for i in range(len(X)):
        X[i] -= np.min(X[i])
        X[i] /= (np.max(np.abs(X[i])) + 1.0)
        
def geometric_mean_preds(_preds):
    preds = _preds.copy()
    for i in range(1, preds.shape[0]):
        preds[0] = np.multiply(preds[0], preds[i])
    return np.power(preds[0], 1/preds.shape[0])

raw_frames = queue.Queue(maxsize=100)
raw_audio_buffer = []
pred_queue = deque(maxlen=10)
def main_process(model, on_predicted):
    
    global raw_audio_buffer
    while not raw_frames.empty():
        raw_audio_buffer.extend(raw_frames.get())
        if len(raw_audio_buffer) >= args.mels_convert_samples: break
    if len(raw_audio_buffer) < args.mels_convert_samples: return
   
    audio_to_convert = np.array(raw_audio_buffer[:args.mels_convert_samples]) / 32767
    raw_audio_buffer = raw_audio_buffer[args.mels_onestep_samples:]
    mels = audio_to_melspectrogram(args, audio_to_convert)
    
    X = []
    for i in range(args.rt_process_count):
        cur = int(i * args.list[1] / args.rt_oversamples)
        X.append(mels[:, cur:cur+args.list[1], np.newaxis])
    X = np.array(X)
    samplewise_normalize_audio_X(X)
    raw_preds = model.predict(X)
    for raw_pred in raw_preds:
        pred_queue.append(raw_pred)
        ensembled_pred = geometric_mean_preds(np.array([pred for pred in pred_queue]))
        on_predicted(ensembled_pred)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def my_exit(model):
    model.close()
    exit(0)
class KerasTFGraph:
    def __init__(self, model_pb_filename, input_name,
                 keras_learning_phase_name, output_name):
        self.graph = load_graph(model_pb_filename)
        self.layer_in = self.graph.get_operation_by_name(input_name)
        self.leayer_klp = self.graph.get_operation_by_name(keras_learning_phase_name)
        self.layer_out = self.graph.get_operation_by_name(output_name)
        self.sess = tf.Session(graph=self.graph)
    def predict(self, X):
        preds = self.sess.run(self.layer_out.outputs[0], 
                              {self.layer_in.outputs[0]: X,
                               self.leayer_klp.outputs[0]: 0})
        return preds
    def close(self):
        self.sess.close()
def get_model(graph_file):
    model_node = {
        'alexnet': ['import/conv2d_1_input',
                    'import/batch_normalization_1/keras_learning_phase',
                    'import/output0']
        
    }
    return KerasTFGraph(
        args.runtime_model_file if graph_file == '' else graph_file,
        input_name=model_node[args.model][0],
        keras_learning_phase_name=model_node[args.model][1],
        output_name=model_node[args.model][2])

def run_predictor():
    model = get_model(graph_file)
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=args.rate,
                input=True,
                
                frames_per_buffer=1024,
                start=False,
                stream_callback=callback 
            )
    
    stream.start_stream()
    while stream.is_active():
        main_process(model, on_predicted)
        time.sleep(1)
    stream.stop_stream()
    stream.close()
    
    audio.terminate()
    my_exit(model)


run_predictor()
