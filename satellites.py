import h5py
import ephem
import pickle
from datetime import datetime, timedelta
import calendar
from ch_util import tools
from ch_util import ni_utils
from ch_util import andata

from numpy import *
from scipy import constants
from scipy import stats

#Define a chime observer for pyephem
chime = ephem.Observer()
chime.lat = '49:19:15.6'
chime.lon = '-119:37:26.4'
chime.elevation = 545

class TransitPhase(object):
    '''
    Run method of this class finds the expected phase at transit
    Parameters: body, data, baselines, observer, name
    body: pyephem object body to find the phase of
    data: andata.Reader object containing the phases in question
    baselines: antenna pairs to use
    name: (optional) overwrite name of ephemeris object (used because the ch_util objects all are named 'None')
    '''
    def __init__(self, body, data, baselines = None, observer = chime, name = None):
        self.set_up = 0
        self.body = body
        self.data = data
        self.observer = observer
        self.bl = baselines
        self.phase = 0
        self.read_data = []
        if not name is None:
            self.name = name

    def transit_time(self, day_num=0):
        '''
        Finds the datetime and unix timestamp of the first transit of the body, and assigns them to self.dt and self.ts.
        '''
        starttime = self.data.time[0]
        # Set time of CHIME observer
        self.observer.date = ephem.Date(datetime.utcfromtimestamp(starttime)+timedelta(days=day_num)) # UTC time supplied to pyephem
        self.body.compute(self.observer)

        self.dt = self.observer.next_transit(self.body)
        offset = self.dt - ephem.Date(datetime.utcfromtimestamp(starttime))
        self.dt = self.dt.datetime()

        self.ts = starttime + (offset / ephem.second)

    def read_transit_data(self, freq=None):
        t_bin = argmin(abs(self.data.time - self.ts))
        sdata = andata.Reader(self.data.files[t_bin//1024])
        if freq is not None:
            sdata.select_freq_physical(freq)
        sdata.select_time_range(self.ts - 10*60 if (self.ts - 10*60) > self.data.time[0] else self.data.time[0], self.ts + 10*60)
        if self.bl is not None:
            sdata.select_prod_pairs(self.bl)
        self.read_data = ni_utils.process_synced_data(sdata.read())
        
    def transit_phase(self):
        transit = argmin(abs(self.read_data.time-self.ts))
        self.phase = angle(self.read_data.vis[:,:,transit])
        self.phases = rollaxis(angle(self.read_data.vis[:,:, transit-1:transit+2]),2)
    
    def max_phase(self):
        guess = argmin(abs(self.read_data.time-self.ts))
        transit = argmax(sum(abs(self.read_data.vis[:,:,guess-10:guess+11]),axis=1),axis=1) + guess - 10
        self.phase_from_max = angle(self.read_data.vis[:,:,transit])
        self.phase_max_f = angle(self.read_data.vis[xrange(self.read_data.vis.shape[0]),:,transit])

    def get_bl(self, ant_0, ant_1):
        feeds = []
        self.phys_bl = []
        inputs = zip(*self.read_data.index_map['input'])[1]
        for i in self.layout:
            if i.input_sn in (inputs[ant_0], inputs[ant_1]):
                feeds.append(i)
                if ant_0 == ant_1:
                    feeds.append(i)
        pos0, pos1 = tools.get_feed_positions(feeds)
        bl2d = pos1-pos0
        bl = array((bl2d[0],bl2d[1],0.0))
        return bl
    
    def read_bl_from_file(self, filename, ant_0, ant_1):
        feeds = []
        inputs = zip(*self.read_data.index_map['input'])[1]
        for i in pickle.load(open(filename)):
            if i.input_sn in (inputs[ant_0], inputs[ant_1]):
                feeds.append(i)

        pos0, pos1 = tools.get_feed_positions(feeds)
        bl2d = pos1-pos0
        bl = array((bl2d[0],bl2d[1],0.0))
        return bl
    
    def run(self, day_num = 0, freq = None):
        self.set_up = True
        self.transit_time(day_num)
        self.read_transit_data(freq)
        self.transit_phase()
        self.max_phase()

    def altaz_to_rec(self, alt, az):
        return array((sin(az)*cos(alt),cos(az)*cos(alt),sin(alt)))
    
    def transit_coords(self):
        starttime = self.data.time[0] # Start of data acquisition

        # Set time of CHIME observer
        self.observer.date = ephem.Date(datetime.utcfromtimestamp(starttime)) # UTC time supplied to pyephem
        self.body.compute(self.observer)

        transit = self.observer.next_transit(self.body)
        self.observer.date = transit
        self.body.compute(self.observer)
        return self.altaz_to_rec(self.body.alt, self.body.az)
    
    def all_coords(self):
        coords= []
        for t in self.read_data.time:
            self.observer.date = ephem.Date(datetime.utcfromtimestamp(t))
            self.body.compute(self.observer)
            coords.append(self.altaz_to_rec(self.body.alt, self.body.az))
        return array(coords)
    
    def expected_phase(self):
        output = []
        self.layout = array(tools.get_correlator_inputs(self.dt))
        if not self.set_up:
            self.transit_time()
            self.read_transit_data()
        if self.bl is not None:
            for baseline in self.bl:
                bl_vector = self.get_bl(*baseline)
                freqs = array([i[0] for i in self.data.freq])
                bdots = dot(self.transit_coords(), bl_vector)
                output.append(2*constants.pi*bdots*(freq*10**6)/constants.c)
        else:
            for ii in xrange(256):
                for jj in xrange(i,256):
                    bl_vector = self.get_bl(ii,jj)
                    freqs = array([i[0] for i in self.data.freq])
                    bdots = dot(self.transit_coords(), bl_vector)
                    output.append(2*constants.pi*bdots*(freqs*10**6)/constants.c)
        return output
    
    def pass_phases(self):
        output = []
        self.layout = array(tools.get_correlator_inputs(self.dt))
        if not self.set_up:
            self.transit_time()
            self.read_transit_data()
        if self.bl is not None:
            for baseline in self.bl:
                bl_vector = self.get_bl(*baseline)
                freqs = [i[0] for i in self.data.freq]
                bdots = dot(self.all_coords(), bl_vector)
                for f in freqs:
                    output.append(2*constants.pi*bdots*(f*10**6)/constants.c)
        else:
            for ii in xrange(256):
                for jj in xrange(ii,256):
                    bl_vector = self.get_bl(ii,jj)
                    freqs = [i[0] for i in self.data.freq]
                    bdots = dot(self.all_coords(), bl_vector)
                    for f in freqs:
                        output.append(2*constants.pi*bdots*(f*10**6)/constants.c)
        return output

    def expected_phase_fbl(self, filename):
        output = []
        if not self.set_up:
            self.transit_time()
            self.read_transit_data()
        for baseline in self.bl:
            bl_vector = self.read_bl_from_file(filename, *baseline)
            freqs = array([i[0] for i in self.data.freq])
            bdots = dot(self.transit_coords(), bl_vector)
            output.append(2*constants.pi*bdots*(freqs*10**6)/constants.c)
        return output


class SatellitePhase(TransitPhase):
    '''Subclass of TransitPhase that gets the phase of a satellite instead of a fixed source'''
    def __init__(self, tle, data, baselines = None , observer = chime, name = None):
        self.set_up = 0
        self.body = ephem.readtle(*tle)
        self.data = data
        self.observer = observer
        self.bl = baselines
    
        if not name is None:
            self.name = name
            
        self.observer.date = ephem.Date(datetime.utcfromtimestamp(data.time[0])) # UTC time supplied to pyephem

#    def find_tle(self):
#        s_list = read_tle_web('http://www.celestrak.com/NORAD/elements/amateur.txt')
#        out_list = find_transits(s_list, datetime.utcfromtimestamp(data.time[0]), datetime.utcfromtimestamp(data.time[-1]))
#        trans = []
#        for sat in out_list:
#            if 'CO-65' in sat[0].name:
#                trans.append(sat[1])
#                self.tle = sat[0]
#        passes = readpass(trans, (401,440))
    def transit_time(self, day_num=0):
        starttime = self.data.time[0]
        # Set time of CHIME observer
        self.body.compute(self.observer)
        sat_pass = self.observer.next_pass(self.body)
        self.observer.date = sat_pass[2]
        self.riset = sat_pass[0]
        self.sett = sat_pass[-2]
        offset = self.observer.date - ephem.Date(datetime.utcfromtimestamp(starttime))
        self.dt = self.observer.date.datetime()
        self.ts = starttime + (offset / ephem.second)
        
    def read_transit_data(self, freq=None):
        t_bin = argmin(abs(self.data.time - self.ts))
        sdata = andata.Reader(self.data.files[t_bin//1024])
        if freq is not None:
            sdata.select_freq_physical(freq)
        rise_ts = calendar.timegm(self.riset.tuple()) if self.riset < self.sett else self.data.time[0]
        sdata.select_time_range(rise_ts, calendar.timegm(self.sett.tuple()))
        if self.bl is not None:
            sdata.select_prod_pairs(self.bl)
#         self.read_data = ni_utils.process_synced_data(sdata.read())    
        self.read_data = sdata.read()

    def all_coords(self,offset=(0,0)):
        coords= []
        for t in self.read_data.time:
            self.observer.date = ephem.Date(datetime.utcfromtimestamp(t))
            self.body.compute(self.observer)
            coords.append(self.altaz_to_rec(self.body.alt+offset[0], self.body.az+offset[1]))
        return array(coords)
        
    def sat_phase(self,freq,offset=(0,0)):
        self.layout = array(tools.get_correlator_inputs(self.dt))
        output = []
        for baseline in self.bl:
            bl_vector = self.get_bl(*baseline)
            bdots = dot(self.all_coords(offset), bl_vector)
            output.append(2*constants.pi*bdots*(freq*10**6)/constants.c)
        return output

    def show_path(self):
        coords= []
        for t in self.read_data.time:
            self.observer.date = ephem.Date(datetime.utcfromtimestamp(t))
            self.body.compute(self.observer)
            coords.append((self.body.alt, self.body.az))
        return coords

    def show_path_amp(self, channel = 0, freq = 928):
        coords= []
        for t in self.read_data.time:
            self.observer.date = ephem.Date(datetime.utcfromtimestamp(t))
            self.body.compute(self.observer)
            coords.append((self.body.alt, self.body.az, abs(self.read_data.vis[freq, channel, where(self.read_data.time == t),])))
        return coords

    def next(self):
        self.observer.date += .005
        return self.run()
