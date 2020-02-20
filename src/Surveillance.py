# GPT-2
import json
import os
import numpy as np
import tensorflow as tf
import re
import configparser

import model, sample, encoder

# Discord
import discord
from discord.ext import commands

config = configparser.ConfigParser()
config.read('settings.ini')

# GPT-2 Generation
def generate(prompt):
    config.read('settings.ini') # Check each run to allow for editing parameters while live

    ##### USER-MODIFIABLE PARAMETERS #####
    model_name = str(config['gpt-2']['model_name'])
    if(str(config['gpt-2']['seed']) == 'None'):
       seed = None
    else: seed = int(config['gpt-2']['seed'])
    if(str(config['gpt-2']['length']) == 'None'):
       length = None
    else: length = int(config['gpt-2']['length'])
    temperature = float(config['gpt-2']['temperature'])
    top_k = int(config['gpt-2']['top_k'])
    ######################################

    ##### FIXED PARAMETERS #####
    # model_name = '355M'
    # seed = None
    # length = None
    # temperature = 0.7
    # top_k = 40
    nsamples = 1
    batch_size = 1
    top_p = 1
    models_dir = 'models'
    ############################
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # "New" Stuff
        context_tokens = enc.encode(prompt)
        generated = 0
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        text = enc.decode(out[0])
        return text

# Discord Command(s)
bot = commands.Bot(command_prefix = str(config['discord']['prefix']))
bot.remove_command("help")

@bot.command()
async def speak(ctx, arg1, arg2=''):
    async with ctx.channel.typing():

        config.read('settings.ini') # Check each run to allow for editing parameters while live

        length = int(config['discord']['length'])
        schizolength = int(config['discord']['schizolength'])

        if (arg1 == "schizo" and arg2 != ''):
            message = str(generate(arg2))
            message = re.sub('(<\|endoftext\|>)', ' ', message) # Regex to remove this commonly occurring garbage
            message = re.sub('^[^a-zA-Z]*', ' ', message) # Remove non-letter junk from beginning of generation. Can cause lopsided quotations in edge-cases
            message = message[:schizolength]
        else:
            message = generate(arg1)
            message = re.sub('(<\|endoftext\|>)', ' ', message) # Regex to remove this commonly occurring garbage
            message = re.sub('^[^a-zA-Z]*', ' ', message) # Remove non-letter junk from beginning of generation. Can cause lopsided quotations in edge-cases
            message = message[:length]
            message = re.sub('(?<=\.)[^.]*\Z', '', message) # Regex to remove anything after the last period, if applicable

        await ctx.send(message)

@bot.command()
async def help(ctx):

    config.read('settings.ini') # Check each run to allow for editing parameters while live
    prefix = str(config['discord']['prefix'])

    embed = discord.Embed(title="Commands", description="All commands use the ` prefix")
    embed.add_field(name=prefix+"help", value="Shows this message!", inline=False)
    embed.add_field(name=prefix+"speak \"[prompt]\"", value="Generates GPT-2 output from the given prompt.", inline=False)
    embed.add_field(name=prefix+"speak schizo \"[prompt]\"", value="Generates fragmented and sometimes incoherent GPT-2 output from the given prompt.", inline=False)
    await ctx.send(embed=embed)

bot.run('YOUR_TOKEN_HERE')
