import uhd
import numpy as np
import argparse
import logging
import yaml
# import threading
# import queue
import time
# import multiprocessing
import os
# import sys
from datetime import datetime
# import subprocess

def write_to_file(queue):
    # logger = create_logger('save')
    while True:
        data = queue.get()
        if (data == "STOP"):
            break
        if (data is not None): 
            # ctime = datetime.now()
            # logger.info('Starting save')
            # print('starting save')
            # np.save(f"/media/roey/T7/{ctime}.npy", data)
            print('got samples')
            # print('ended save')
            # logger.info('Ended Save')
        time.sleep(1)

def save_to_file(name, value):
    value.tofile(name)

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
    return(configuration)

def log_padded_info(action, value_name, value, logger, padding):
    logger.info(action.ljust(padding, " ") + value_name.ljust(padding, " ") + str(value))

def transmit_sample(usrp, signal, sample_rate, center_freq, gain, logger, padding):
    # Configuring USRP
    if (sample_rate == usrp.get_tx_rate()):
        log_padded_info("Use", "tx_rate", usrp.get_tx_rate(), logger, padding)
    else:
        log_padded_info("Set", "tx_rate", sample_rate, logger, padding)
        usrp.set_tx_rate(sample_rate)
        log_padded_info("Get", "tx_rate", usrp.get_tx_rate, logger, padding)

    # logger.info(f'{"Setting"<:padding} {"tx_freq"<:padding} : {uhd.types.TuneRequest(center_freq)}')
    # usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
    # logger.info(f'{"Got"<:padding} {"tx_freq"<:padding} : {usrp.get_tx_freq()}')
    if (uhd.types.TuneRequest(center_freq) == usrp.get_tx_freq()):
        log_padded_info("Use", "tx_freq", usrp.get_tx_freq(), logger, padding)
    else:
        log_padded_info("Set", "tx_freq", center_freq, logger, padding)
        usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
        log_padded_info("Get", "tx_freq", usrp.get_tx_freq(), logger, padding)


    # logger.info(f'{"Setting"<:padding} {"gain"<:padding} : {gain}')
    # usrp.set_tx_gain(gain)
    #   logger.info(f'{"Got"<:padding} {"gain"<:padding} : { usrp.get_tx_gain()}')
    if (gain == usrp.get_tx_gain()):
        log_padded_info("Use", "tx_gain", usrp.get_tx_gain(), logger, padding)
    else:
        log_padded_info("Set", "tx_gain", gain, logger, padding)
        usrp.set_tx_gain(gain)
        log_padded_info("Get", "tx_gain", usrp.get_tx_gain(), logger, padding)

    # Create tx stream
    tx_stream = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

    # Stream buffer
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = False
    samples_per_buffer = tx_stream.get_max_num_samps()
    total_samples = len(signal)
    logger.info(f"Samples per buffer: {samples_per_buffer}, total samples: {total_samples}")

    # return #################################################################################

    # signal = [signal]
    logger.info(f"Starting transmission - Center: {center_freq}, Sample rate: {sample_rate}, Bandwidth: {usrp.get_tx_bandwidth()}")
    for i in range(0, total_samples, samples_per_buffer):
        buffer = signal[i:i+samples_per_buffer]
        tx_stream.send([buffer], metadata)
        metadata.start_of_burst = False
    metadata.end_of_burst = True
    logger.info(f"Ended transmission - Center: {center_freq}, Sample rate: {sample_rate}")
    # tx_stream.send(np.array([], dtype=np.complex64), metadata)

    # print("Transmission completed")


def gen_samp(frequency, duration, rate):
    num_samps = int(rate * duration)
    t = np.arange(num_samps) / rate
    signal = np.exp(2j * np.pi * frequency * t)
    return signal

def move_file(file_to_move, destination):
    #proc = multiprocessing.Process(target=os.system, args=(f"mv {file_to_move} {destination}",))
    #proc.start()
    os.system(f"mv {file_to_move} {destination}")

def main(queue, logger, padding):
    args = parse_args()
    options = read_yaml(args.configuration)
    antenna = "TX/RX"
    usrp = uhd.usrp.MultiUSRP()
    
    bw = options['rate'] # Bandwidth is the same as the sampling rate

    log_padded_info("Set", "rx_antenna", antenna, logger, padding)
    usrp.set_rx_antenna(antenna, 0)
    log_padded_info("Get", "rx_antenna", usrp.get_rx_antenna(), logger, padding)

    log_padded_info("Set", "tx_antenna", antenna, logger, padding)
    usrp.set_tx_antenna('TX/RX', 0)
    log_padded_info("Get", "tx_antenna", usrp.get_tx_antenna(), logger, padding)

    log_padded_info("Set", "rx_bandwidth", bw, logger, padding)
    usrp.set_rx_bandwidth(bw)
    log_padded_info("Get", "rx_bandwidth", usrp.get_rx_bandwidth(), logger, padding)

    log_padded_info("Set", "tx_bandwidth", bw, logger, padding)
    usrp.set_tx_bandwidth(bw)
    log_padded_info("Get", "tx_bandwidth", usrp.get_tx_bandwidth(), logger, padding)

    num_samps = int(np.ceil(options['rx_duration'] * options['rate']))
    if not isinstance(options['channels'], list):
        options['channels'] = [options['channels']]
    
    # cf = 2.45e9 # Center Frequency
    # gain = 60.0 
    # sr = 1e6 # Sample Rate
    # signal = np.exp(2j * np.pi * 1000 * np.arange(10000) /sr)
    good_signal = gen_samp(options['tx_rate'], options['tx_duration'], options['tx_rate']) 

    for i in range(10):
        filename = f"/media/roey/T7/{i}.npy"
        logger.info(f"Strating reception of {num_samps} samples on {options['rx_freq']} frequency in {options['rate']} sample rate and {options['rx_gain']} gain")
        samps = usrp.recv_num_samps(num_samps, options['rx_freq'], options['rate'], [0], options['rx_gain'])
        logger.info(f"Finished reception")
        logger.info(f"Starting the save of {filename}")
        np.save(f"/media/roey/T7/{i}.npy", samps)
        logger.info(f"Finished Saving")
        logger.info(f"Starting transmission at {options['tx_freq']}Hz with rate of {options['rate']}Hz and gain of {options['tx_gain']} for {options['tx_duration']} secs")
        # transmit_sample(usrp, good_signal, options['rate'], options['tx_freq'], options['tx_gain'], logger, padding)
        usrp.send_waveform(good_signal, options['tx_duration'], options['tx_freq'], options['tx_rate'], [0], options['tx_gain'])
        logger.info(f"Ended transmission")



    # signal = gen_samps_for_tx(1, 1e6)

    # transmit_sample(usrp, signal2, sr, cf, gain)
    # for i in range(10):
    #     base_file_name = f"{i:04d}.npy"
    #     temp_file_name = os.path.join("tmp", base_file_name)
    #     dest_file_name = os.path.join("/media/roey/T7", base_file_name)
    # #     recv_buffer = np.memmap(temp_file_name, dtype=np.complex64, mode="w+", offset=0, shape=(num_buffs, num_channels, max_samples_per_packet))
    # #     for i in range(num_buffs):
    # #         # print(i)
    # #         rx_streamer.recv(recv_buffer[i] , metadata) # TODO: Add metadata variable
    #     logger.info(f"Strating reception of {num_samps} samples on {options['rx_freq']} frequency in {options['rate']} sample rate and {options['rx_gain']} gain")
    #     samps = usrp.recv_num_samps(num_samps, options['rx_freq'], options['rate'], [0], options['rx_gain'])
    #     logger.info(f"Finished reception")
    #     logger.info(f"Saving")
    #     print('saving')
    #     np.save(temp_file_name,samps)
    #     print('haha saving')
    #     move_file(temp_file_name, dest_file_name)
    #     print('haha moved')
    #     #queue.put(samps)
    #     logger.info(f"Finished save")
    #     # transmit_sample(usrp, signal2, sr, cf, gain, logger, padding)
    
    # samps = usrp.recv_num_samps(num_samps, options['rx_freq'], options['rate'], [0], options['rx_gain'])
    # logger.info("started saving")
    # samps.tofile(options['output_file'])
    # logger.info("endded saving")



    # save_t = threading.Thread(target=save_to_file, args=(options['output_file'], samps))
    # save_t.start()
    # samps = usrp.recv_num_samps(num_samps, options['rx_freq'], options['rate'], [0], options['rx_gain'])
    # print("done")
    # save_t.join()
    # print("saved")

    # st_args = uhd.usrp.StreamArgs("fc32", "sc16")

    # usrp.set_rx_spp(2**16)
    # rx_streamer = usrp.get_rx_stream(st_args)

    # usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(options['rx_freq']))
    # usrp.set_rx_rate(options['rate'])
    # usrp.set_rx_gain(options['rx_gain'])

    # num_channels = rx_streamer.get_num_channels()
    # max_samples_per_packet = rx_streamer.get_max_num_samps()
    # print(max_samples_per_packet)
    # print('ffff')
   
    # metadata = uhd.types.RXMetadata()

    # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    # stream_cmd.stream_now = (num_channels == 1)
    # stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + 0) # Change the number for Init Delay
    # rx_streamer.issue_stream_cmd(stream_cmd)

    # actual_rate = usrp.get_rx_rate()
    # num_buffs = int(options['rx_duration'] * options['rate'] / max_samples_per_packet)
    # for i in range(10):
    #     base_file_name = f"{i:04d}.npy"
    #     temp_file_name = os.path.join("tmp", base_file_name)
    #     dest_file_name = os.path.join("/media/roey/T7", base_file_name)
    #     recv_buffer = np.memmap(temp_file_name, dtype=np.complex64, mode="w+", offset=0, shape=(num_buffs, num_channels, max_samples_per_packet))
    #     for i in range(num_buffs):
    #         # print(i)
    #         rx_streamer.recv(recv_buffer[i] , metadata) # TODO: Add metadata variable
    #     recv_buffer.flush()
    #     del recv_buffer
    #     print('hi hi hi')
    #     #move_file(temp_file_name, dest_file_name)


    # print(num_channels)
    # recv_buffer = np.empty(())

    # os._exit(0)

    # Receive samples
    # print('starting with clock ', options['rate'])
    # samps = usrp.recv_num_samps(num_samps, options['rx_freq'], options['rate'], [0], options['rx_gain'])
    # print('finished, starting write')
    # with open(options['output_file'], 'wb') as f:
        #  np.save(f, samps, allow_pickle=False, fix_imports=False)
    # print('done!')

    #Send samples
    #TODO
    # usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)

    #for reference
    # samps.astype(np.complex64).tofile(fname)

def create_logger(name):
    # Creating an object
    logger = logging.getLogger(name)
    
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(f'{name}.log', 'w')
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        # f"{'%(asctime)s | %(levelname)s':<{30}} - %(message)s"
        '%(asctime)-12s | %(levelname)-6s - %(message)s'
    )

    fh.setFormatter(fmt)

    logger.addHandler(fh)

    return logger

if __name__ == "__main__":
    # data_queue = multiprocessing.Queue()

    # proc = multiprocessing.Process(target=write_to_file, args=(data_queue,))
    # proc.start()

    mylogger = create_logger(__name__)
    padding = 12
    main(1, mylogger, padding)
    # data_queue.put("STOP")
    
    # num_elements = int(4.5 * 1024**3 // 8)
    # heavy_arr = np.random.rand(num_elements)

    # array_size_bytes = heavy_arr.nbytes 
    # print(array_size_bytes / 1024**2)

    # np.save('/media/roey/T7/test.npy', heavy_arr)


# memorymap to tempfile
# save to tempfile on memory (not mmap) and call a subproccess to save it, 