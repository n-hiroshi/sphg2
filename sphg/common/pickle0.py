"""
Utilify function to handle pickle files
"""
import pickle
import os

def pickle_func(picklefile,func,*args):
        if not (os.path.isfile(picklefile)):
                returnlist = func(*args)
                with open(picklefile, mode='wb') as f:
                    #pickle.dump(returnlist, f)
                    pickle.dump(returnlist, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
        else:
                with open(picklefile, mode='rb') as f:
                    #returnlist = pickle.load(f)
                    returnlist = pickle.load(MacOSFile(f))
        return returnlist

def pickle_obj(picklefile,obj):
        if not (os.path.isfile(picklefile)):
                with open(picklefile, mode='wb') as f:
                    #pickle.dump(obj, f)
                    pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
        else:
                with open(picklefile, mode='rb') as f:
                    #obj = pickle.load(f)
                    obj = pickle.load(MacOSFile(f))
        return obj


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        #print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            #print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            #print("done.", flush=True)
            idx += batch_size

