import discord
import os
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from discord.ext import commands,tasks
import re, string
import scipy.sparse

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

client = commands.Bot(command_prefix = '!')
model_path = 'NB_SVM_model.sav'
vectorizer_path = 'vectorizer.sav'
token = '' # Enter discord bot token here
model = pickle.load(open(model_path, 'rb'))
vec = pickle.load(open(vectorizer_path, 'rb'))

@client.event
async def on_ready():
	print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
	if message.author == client.user:
		return
	check = vec.transform([message.content])
	toxicity = model.predict(check)
	print(toxicity)
	if toxicity[0] == 1:
		await message.channel.send('That\'s not very nice')

client.run(token)