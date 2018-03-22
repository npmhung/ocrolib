import configparser
import os

config = configparser.ConfigParser()
# path = '/media/warrior/MULTIMEDIA/' \
#        'Newworkspace/Nissay/output/run_form_cut_4_template/template_dump/meta/temp3/'
path = '/media/warrior/MULTIMEDIA/Newworkspace/release/flex_scan_engine/data/meta/temp1/'
#path = '/media/warrior/MULTIMEDIA/' \
#       'Newworkspace/Nissay/output/run_form_cut_4_template/template_dump_new/meta/temp4/'

# path = "/home/taprosoft/Downloads/test_segmented/nissay/run/meta/temp1/"

# config.add_section('rotation')
# config.add_section('number')
# config.add_section('regiontype')
# config.add_section('connectedcpn')
# config.add_section('datefield')
# config.add_section('horizontalpivot')
# config.add_section('textnumber')
for file in os.listdir(path):
	filename = '%s%s'%(path, file)
	# print('filename', filename)
	config.read(filename)
	# config.set('reduction', 'threshgapwidth', '30')
	# config.set('reduction', 'recheckpivot', '0')
	# config.set('reduction', 'recheck_area', '100')
	# config.set('reduction', 'threshmid','15')
	# config.set('rotation', 'rotate', '1')
	# config.set('rotation', 'angle', '5')
	# config.set('horizontalpivot','checkcross','0')
	# config.set('horizontalpivot','plot','0')
	# config.set('horizontalpivot', 'bandwidth', '5.0')
	# config.set('horizontalpivot', 'bandwidth', '5.0')
	# config.set('checksmoothhist', 'threshCoef', '20')
	# config.set('rotation', 'top', '30')
	# config.set('rotation', 'bottom', '30')
	# config.set('rotation', 'left', '5')
	# config.set('rotation', 'right', '5')
	# config.set('checksmoothhist','cf_interval', '20')
	# config.set('wordCut', 'resizecoef', '80')
	# config.set('enhance', 'horcoef', '4')
	# config.set('checkBlank', 'threshsum', '25')
	# config.set('number','isnumbercut','0')
	# config.set('textnumber', 'isnumber','0')
	# config.set('textnumber', 'line1number', '0')
	# config.set('textnumber', 'line2number', '0')
	# config.set('textnumber', 'line3number', '0')
	# config.set('textnumber', 'line4number', '0')
	# config.set('textnumber', 'line5number', '0')
	# config.set('datefield', 'isdate','0')
	# config.set('datefield', 'd','0.23')
	# config.set('datefield', 'm', '0.57')
	# config.set('regiontype', 'type', 'cpn')
	# config.set('connectedcpn', 'remove_area', '50')
	config.set('connectedcpn', 'findcpn','0')
	with open(filename, 'w') as configfile:
		config.write(configfile)