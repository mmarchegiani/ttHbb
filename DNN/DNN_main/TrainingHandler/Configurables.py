import sys

def checkForProperty(config, section, option):
	if not config.has_option(section, option):
		raise ValueError('Property '+option+' is required in the section '+section)

def validate(config):
	checkForProperty(config, 'training', 'model')
	checkForProperty(config,'output','output-folder')

	if config.get('training', 'model') == 'dnn':
		checkForProperty(config,'training','epochs')
		checkForProperty(config,'training','batch-size')
	elif config.get('training', 'model') == 'bdt':
		checkForProperty(config,'training','workers')
		checkForProperty(config,'training','trees')
		checkForProperty(config,'training','max-depth')

	checkForProperty(config,'training','training-labels')
	checkForProperty(config,'input','data-train')
	checkForProperty(config,'input','data-val')
	