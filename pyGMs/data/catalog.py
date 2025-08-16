"""
catalog.py

Defines Models (uai-file oriented problem instances) and Catalogs (collections of Models)
along with a just-in-time mechanism for downloading model files from a repository.

Version 0.3.1 (2025-08-15)
(c) 2015-2025 AlexanderIhler under the FreeBSD license; see license.txt for details.
"""


import tempfile
import json

import os
import sys
import requests
import warnings
from urllib.parse import urlparse

## Global package variables
_tempdir = None
sources = []
cache = None

## Package constants
__SOURCES_FILENAME__ = "sources.json"
__CATALOG_FILENAME__ = "index.json"
__STATS_FILENAME__ = "statistics.csv"


#### Helper functions ##############################################################
def get_urldata(url, verbose=True):
  if (verbose): print(f'Downloading: {url}')
  r = requests.get(url, allow_redirects=False)
  data = r.content
  return data


#### Single Model Object ###############################################################
# Contains reported statistcs about this model, including evidence & elim order, and a path to the file
# alternate method: lazy get file etc when information is accessed? also get() function?

class Model(object):
  def __init__(self,path='',baseurl='', **kwargs):
    self.__dict__ = kwargs
    self.__base__ = path
    self.__baseurl__ = baseurl
    self.__file__ = None
    self.__evidence__ = None
    self.__order__ = None

  def __lazyget(self,filename):
    parsed = urlparse(filename)
    fileurl = filename if bool(parsed.netloc) else os.path.join(self.__baseurl__,filename)
    filedir = os.path.join(self.__base__,os.path.dirname(filename))
    fileall = os.path.join(self.__base__,filename)
    if not os.path.isdir(filedir):
      raise ValueError(f'Models directory {filedir} not found; please initialize.')
    if not os.path.isfile(fileall):
      try:
        data = get_urldata(fileurl, verbose=False)
        with open(fileall,'wb') as fh: fh.write(data)
      except:
        raise ValueError(f'Could not find or download {filename}') 
    return fileall

  @property
  def file(self): 
    if self.__file__ is None: self.__file__ = self.__lazyget(self.modelfile)
    return self.__file__

  @property
  def evidence(self):
    if self.__evidence__ is None:
      if self.num_evid == 0: self.__evidence__ = { }
      else:
        filepath = self.__lazyget(self.evidencefile)
        evid = np.loadtxt(filepath).flatten().astype(int)
        if len(evid)!=2*evid[0]+1: raise ValueError(f'Error / unknown format in {self.evidencefile}')
        self.__evidence__ = {i:x for i,x in zip(evid[1::2],evid[2::2])}
    return self.__evidence__.copy()

  @property
  def order(self):
    if self.__order__ is None:
      filepath = self.__lazyget(self.orderfile)
      order = np.loadtxt(filepath).astype(int).flatten()
      if len(order)!=order[0]+1: raise ValueError(f'Error / unknown format in {self.orderfile}')
      self.__order__ = tuple(order[1:])
    return self.__order__

  def get(self):
    _ = (self.file, self.evidence, self.order)

########################################################################################

class Catalog(object):
    """Recursive catalog of model files in current cache or available at known repositories"""
    def __init__(self, cache=None, path=None, source=None, verbose=False):
        self.cache = cache
        self.path = '' if path is None else path
        self.sources = [] if source is None else [source]
        self.verbose = verbose
    ## Somewhat unnecessary functions for manipulating web-based data resource locations
    def clear_sources(self):
        self.sources = []
    def add_source(self, url):
        self.sources += [url]
    def add_source_file(self, filename):
        with open(filename) as fh: data = json.load(fh)
        self.sources += [d['url'] for d in data.values() if 'url' in d and d['url'] is not None]
    ## Set the cache directory, optionally using a temporary directory
    def set_cache(self, path=None, create=False, update=True):
        """Set the location for local caching of model files and statistics.
        path   (str)  : directory path to use for caching. 'None' will create a temporary directory.
        create (bool) : create path if it does not exist
        update (bool) : create cached model catalog from sources
        """
        if path is None:
            global _tempdir
            if _tempdir is None: _tempdir = tempfile.TemporaryDirectory()
            path = _tempdir.name
        if not os.path.isdir(path):
            if not create:
                raise ValueError(f'Failed to find cache directory {path}! Use set_cache(None) for a temporary cache, or create=True to create.')
            else:
                os.makedirs(path, exist_ok=True)
        self.cache = path
        if update and not os.path.isfile(self.__incache(__CATALOG_FILENAME__)): self.update_cache()

    def update_cache(self, path=None):
      """Update list of modelset packages from known repositories. Specify 'path' to override current cache location."""
      if path is None: path = self.cache
      all_modelsets = {}
      if self.verbose: print('Initializing cache index from sources:')
      for url in self.sources:
        try:
          if self.verbose: print(f'  {url}: ',end='')
          src_data = get_urldata(url, verbose=False)
          src_models = json.loads(src_data)
        except:
          if self.verbose: print(f' failed; skipping!')
          continue
        if self.verbose: print(f'done!')
        all_modelsets.update( src_models )
      try:
        with open(self.__incache(__CATALOG_FILENAME__),'w') as fh:
          json.dump(all_modelsets,fh,indent=4)
      except:
        warnings.warn('Unable to write model catalog! Please check cache directory permissions.')

    def __incache(self,*args):
        """Join the function arguments with the cache location to provide an absolute path"""
        if self.cache is None: self.set_cache()
        return os.path.join(self.cache,self.path,*args)

    ##### Drilling down to get sets and subsets of models #####################################
    def __getitem__(self, key):
        ## If longer path, recursively move through sub-paths:
        if '/' in key:              
            current = self
            parts = key.split('/')
            for p in parts: current = current[p]
            return current
        ## If our cache has a catalog file, follow along:
        next_url = None
        if os.path.exists( self.__incache(__CATALOG_FILENAME__) ):
            with open(self.__incache(__CATALOG_FILENAME__)) as fh: sets = json.load(fh)
            if key in sets: next_url = os.path.dirname(sets[key]['modelset'])
        ## If already-cached subdirectory, just move down to that level
        if os.path.isdir( self.__incache(key) ):
            return Catalog( self.cache, os.path.join(self.path,key), next_url)
        elif os.path.exists( self.__incache(__STATS_FILENAME__) ):
            import pandas as pd
            stats = pd.read_csv(self.__incache(__STATS_FILENAME__),delimiter=',',header=0,skipinitialspace=True)
            ## Load stats file into indexable structure
            try:
                info = stats.set_index('name').loc[key] 
                return Model(self.cache, self.sources[0], **dict(info))
            except:
                raise ValueError('Model {key} not found in statistics file!')
        elif os.path.exists( self.__incache(__CATALOG_FILENAME__) ):
            ## Already cached catalog file (models or modelsets); load and check
            with open(self.__incache(__CATALOG_FILENAME__)) as fh: 
                sets = json.load(fh)
            if key not in sets: raise ValueError(f'Set {key} not found in cache at {self.__incache(__CATALOG_FILENAME__)}')
        else:
            sets = {}
            for url in self.sources:
                try:
                    if not url[-len(__CATALOG_FILENAME__):]==__CATALOG_FILENAME: url=os.path.join(url,__CATALOG_FILENAME__)
                    urldata = get_urldata(url, verbose=self.verbose)
                    sets.update( json.loads(urldata) )
                except: pass
            if len(sets):
                with open(self.__incache(__CATALOG_FILENAME__),'wb') as fh: json.dump(sets,fh,indent=4)
            if key not in sets: raise ValueError(f'Set {key} not found at URLs {self.sources}')

        os.makedirs(self.__incache(key), exist_ok=True)
        next_url = sets[key]['modelset']
        urldata = get_urldata(next_url, verbose=self.verbose)
        with open(self.__incache(key,os.path.basename(next_url)),'wb') as fh: fh.write(urldata);
        return Catalog( self.cache, os.path.join(self.path,key), next_url )

    def keys(self):
        if os.path.exists( self.__incache(__STATS_FILENAME__) ):
            import pandas as pd
            stats = pd.read_csv(self.__incache(__STATS_FILENAME__),delimiter=',',header=0,skipinitialspace=True)
            return list(stats['name'])
        ## If not a set of sets:        
        cached = [d for d in os.listdir(self.cache) if os.path.isdir(d)]
        if os.path.exists( self.__incache(__CATALOG_FILENAME__) ):
            with open(self.__incache(__CATALOG_FILENAME__)) as fh: 
                known = json.load(fh).keys()
        else:
            sets = {}
            for url in self.sources:
                try:
                    if not url[-len(__CATALOG_FILENAME__):]==__CATALOG_FILENAME: url=os.path.join(url,__CATALOG_FILENAME__)
                    urldata = get_urldata(url, verbose=self.verbose)
                    sets.update( json.loads(urldata) )
                except: pass
            known = sets.keys()        
        return sorted(list(set(known)|set(cached)))

    def items(self):
        for key in self.keys(): yield key,self[key]

    def __iter__(self):
        for key in self.keys(): yield key


