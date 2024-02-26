import uhd
import numpy as np
import argparse
import logging
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", type=str, required=False, default="default_configuration.yaml") 
    # parser.add_argument("-a", "--args", default="", type=str)
    # parser.add_argument("-o", "--output-file", type=str, required=True)
    # parser.add_argument("-f", "--freq", type=float, required=True)
    # parser.add_argument("-r", "--rate", default=1e6, type=float)
    # parser.add_argument("-d", "--duration", default=5.0, type=float)
    # parser.add_argument("-c", "--channels", default=0, nargs="+", type=int)
    # parser.add_argument("-g", "--gain", type=int, default=10)
    return parser.parse_args()

def gen_samps_for_tx(tx_duration, rate):
	n_samps = int(tx_duration * rate)
	return np.exp(1j * np.random.normal(0, 1, n_samps))
	
def read_yaml(filename):
	with open(filename, 'r') as file:
		configuration = yaml.safe_load(file)
	return configuration
	print(configuration)

def main():
    args = parse_args()
    print(args.configuration)
    return
    options = read_yaml("conf.yaml")
    usrp = uhd.usrp.MultiUSRP(args.args)
    num_samps = int(np.ceil(args.duration*args.rate))
    if not isinstance(args.channels, list):
        args.channels = [args.channels]
        
    #Receive samples
    samps = usrp.recv_num_samps(num_samps, args.freq, args.rate, 0, args.gain)
    with open(args.output_file, 'wb') as f:
        np.save(f, samps, allow_pickle=False, fix_imports=False)
        
    #Send samples
    #TODO
    # usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
    
    #for reference
    # samps.astype(np.complex64).tofile(fname)

if __name__ == "__main__":
    main()
