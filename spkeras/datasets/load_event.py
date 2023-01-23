class load_event(object):
    def __init__(self,dataset='cifar10_dvs', **kwargs):
        if dataset == 'cifar10_dvs':
            self.fun = self.loadaerdat
        elif dataset == 'dvs128_gesture':
            self.fun = self.npload
        elif dataset == 'n_caltech101':
            self.fun = self.read_dataset
        else:
            raise Exception("Only supports dataset: cifar10_dvs, dvs128_gesture or n_caltech101")
    def __call__ (self,path):
        return self.fun(path)
        
    def npload(self,path):
        import numpy as np
        return np.load(path)
    #modified code, source from https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/__init__.py
    def load_aedat_v3(self,file_name: str):
        
        import struct
        '''
        :param file_name: path of the aedat v3 file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict
        This function is written by referring to https://gitlab.com/inivation/dv/dv-python . It can be used for DVS128 Gesture.
        '''
        with open(file_name, 'rb') as bin_f:
            # skip ascii header
            line = bin_f.readline()
            while line.startswith(b'#'):
                if line == b'#!END-HEADER\r\n':
                    break
                else:
                    line = bin_f.readline()

            txyp = {
                't': [],
                'x': [],
                'y': [],
                'p': []
            }
            while True:
                header = bin_f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                e_type = struct.unpack('H', header[0:2])[0]
                e_source = struct.unpack('H', header[2:4])[0]
                e_size = struct.unpack('I', header[4:8])[0]
                e_offset = struct.unpack('I', header[8:12])[0]
                e_tsoverflow = struct.unpack('I', header[12:16])[0]
                e_capacity = struct.unpack('I', header[16:20])[0]
                e_number = struct.unpack('I', header[20:24])[0]
                e_valid = struct.unpack('I', header[24:28])[0]

                data_length = e_capacity * e_size
                data = bin_f.read(data_length)
                counter = 0

                if e_type == 1:
                    while data[counter:counter + e_size]:
                        aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                        timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                        x = (aer_data >> 17) & 0x00007FFF
                        y = (aer_data >> 2) & 0x00007FFF
                        pol = (aer_data >> 1) & 0x00000001
                        counter = counter + e_size
                        txyp['x'].append(x)
                        txyp['y'].append(y)
                        txyp['t'].append(timestamp)
                        txyp['p'].append(pol)
                else:
                    # non-polarity event packet, not implemented
                    pass
            txyp['x'] = np.asarray(txyp['x'])
            txyp['y'] = np.asarray(txyp['y'])
            txyp['t'] = np.asarray(txyp['t'])
            txyp['p'] = np.asarray(txyp['p'])
            return txyp

    #modified code, source from https://github.com/SensorsINI/processAEDAT/blob/master/jAER_utils/loadaerdat.py
    def loadaerdat(self,datafile='/tmp/aerout.dat', length=0, version="aedat", debug=1, camera='DVS128'):
        import struct
        import os
        import numpy as np
        V3 = "aedat3"
        V2 = "aedat"  # current 32bit file format
        V1 = "dat"  # old format

        EVT_DVS = 0  # DVS event type
        EVT_APS = 1  # APS event
        """    
        load AER data file and parse these properties of AE events:
        - timestamps (in us), 
        - x,y-position [0..127]
        - polarity (0/1)
        @param datafile - path to the file to read
        @param length - how many bytes(B) should be read; default 0=whole file
        @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
        @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
        @param camera='DVS128' or 'DAVIS240'
        @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
        """
        # constants
        aeLen = 8  # 1 AE event takes 8 bytes
        readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
        td = 0.000001  # timestep is 1us   
        if(camera == 'DVS128'):
            xmask = 0x00fe
            xshift = 1
            ymask = 0x7f00
            yshift = 8
            pmask = 0x1
            pshift = 0
        elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
            xmask = 0x003ff000
            xshift = 12
            ymask = 0x7fc00000
            yshift = 22
            pmask = 0x800
            pshift = 11
            eventtypeshift = 31
        else:
            raise ValueError("Unsupported camera: %s" % (camera))

        if (version == "dat"):
            #print ("using the old .dat format")
            aeLen = 6
            readMode = '>HI'  # ushot, ulong = 2B+4B

        aerdatafh = open(datafile, 'rb')
        k = 0  # line number
        p = 0  # pointer, position on bytes
        statinfo = os.stat(datafile)
        if length == 0:
            length = statinfo.st_size    
        #print ("file size", length)

        # header
        lt = aerdatafh.readline()
        while lt and lt[0] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline() 
            if debug >= 2:
                print (str(lt))
            continue

        # variables to parse
        timestamps = []
        xaddr = []
        yaddr = []
        pol = []

        # read data-part of file
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

        #print (xmask, xshift, ymask, yshift, pmask, pshift)    
        while p < length:
            addr, ts = struct.unpack(readMode, s)
            # parse event type
            if(camera == 'DAVIS240'):
                eventtype = (addr >> eventtypeshift)
            else:  # DVS128
                eventtype = EVT_DVS

            # parse event's data
            if(eventtype == EVT_DVS):  # this is a DVS event
                x_addr = (addr & xmask) >> xshift
                y_addr = (addr & ymask) >> yshift
                a_pol = (addr & pmask) >> pshift

                timestamps.append(ts)
                xaddr.append(x_addr)
                yaddr.append(y_addr)
                pol.append(a_pol)

            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen        
            
        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }    
        timestamps = np.asarray(timestamps)
        start = int(np.argwhere(timestamps == 0)[0])
        txyp['x'] = np.asarray(xaddr[start:])
        txyp['y'] = np.asarray(yaddr[start:])
        txyp['t'] = np.asarray(timestamps[start:])
        txyp['p'] = np.asarray(pol[start:])            
        return txyp
    
    #modified code, source from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    def read_dataset(self,filename):
        """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
        import numpy as np
        f = open(filename, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7 #bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        #Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        #Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]
        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }

        txyp['x'] = np.asarray(all_x)
        txyp['y'] = np.asarray(all_y)
        txyp['t'] = np.asarray(all_ts)
        txyp['p'] = np.asarray(all_p)   

        return txyp
    
    #modified code, source from https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/dvs128_gesture.py
    def split_aedat_files_to_np(self,fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = load_aedat_v3(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
        label_file_num = [0] * 11

        # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
        # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
        # are replaced by new codes.


        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            #t_end  = t_start+1.5*1e6
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, f'{fname}_{label}.npz')

            

            np.savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1