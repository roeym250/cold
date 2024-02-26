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
    print(configuration)
    return configuration

def transmit_sample(usrp, signal, sample_rate, center_freq, gain):
    # Configuring USRP
    usrp.set_tx_rate(sample_rate)
    usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
    usrp.set_tx_gain(gain)

    # Create tx stream
    tx_stream = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

    # Stream buffer
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = False
    samples_per_buffer = tx_stream.get_max_num_samps()
    total_samples = len(signal)

    # signal = [signal]

    for i in range(0, total_samples, samples_per_buffer):
        buffer = signal[i:i+samples_per_buffer]
        tx_stream.send([buffer], metadata)
        metadata.start_of_burst = False
    metadata.end_of_burst = True
    # tx_stream.send(np.array([], dtype=np.complex64), metadata)

    print("Transmission completed")


def gen_samp(frequency, duration, rate):
    num_samps = int(rate * duration)
    t = np.arange(num_samps) / rate
    signal = np.exp(2j * np.pi * frequency * t)
    return signal


def main():
    args = parse_args()
    options = read_yaml(args.configuration)
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_antenna('TX/RX', 0)
    usrp.set_tx_antenna('TX/RX', 0)
    print(usrp.get_rx_antenna())
    num_samps = int(np.ceil(options['duration'] * options['rate']))
    if not isinstance(options['channels'], list):
        options['channels'] = [options['channels']]

    # Receive samples
    # print('starting with clock ', options['rate'])
    # samps = usrp.recv_num_samps(num_samps, options['freq'], options['rate'], [0], options['gain'])
    # print('finished, starting write')
    # with open(options['output_file'], 'wb') as f:
        #  np.save(f, samps, allow_pickle=False, fix_imports=False)
    # print('done!')

    #Send samples
    #TODO
    # usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
    
    cf = 2.45e9 # Center Frequency
    gain = 60 
    sr = 1e6 # Sample Rate
    signal = np.exp(2j * np.pi * 1000 * np.arange(10000) /sr)
    signal2 = gen_samp(cf, 3, sr) 
    # signal = gen_samps_for_tx(1, 1e6)

    # transmit_sample(usrp, signal2, sr, cf, gain)
    for i in range(10):
        samps = usrp.recv_num_samps(num_samps, options['freq'], options['rate'], [0], options['gain'])
        transmit_sample(usrp, signal2, sr, cf, gain)

    #for reference
    # samps.astype(np.complex64).tofile(fname)

if __name__ == "__main__":
    main()
