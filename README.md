# Surveillance
A basic Discord bot with three goals:
1. Generate reasonable output using GPT-2 from guided user input (commands)
2. Generate the above output by reacting to conversation "naturally"
3. Do both of these while being easy to set up and run for non-tech-savvy users

Currently it generates output using GPT-2, but this may change if better generation techniques appear.

### Installation
This project uses Python 3.7.0
First install the requirements, this is done using [pip](https://bootstrap.pypa.io/get-pip.py "pip").
```shell
pip install -r requirements.txt
```
Then download your model of choice. By default, Surveillance is modelled for the 355M model - if you want to use another one, just edit it in Surveillance.py
```shell
python download_model.py 355M
```
Surveillance is now installed. Make sure to change the placeholder token to your bot's token in Surveillance.py
You can find out how to set up the foundation for a bot and get its token [here](https://discordapp.com/developers "here").

### Usage
Once you have added your proper token and run Surveillance.py, your bot will be online and ready to use. From here you can use the following command(s):
```shell
`speak "Input prompt should go here"
```
This will take the  given input (in this case "Input prompt should go here") and generate a response using GPT-2. This may take some time depending on the strength of the device running the bot, so please be patient. Please also keep in mind that the input prompt *must* be contained within quotation marks, or the bot will only attempt to generate output from the first word after the \`speak command.