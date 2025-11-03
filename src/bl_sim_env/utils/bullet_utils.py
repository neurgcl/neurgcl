"""A wrapper for pybullet to manage different clients."""
from __future__ import absolute_import
from __future__ import division
import os
import functools
import inspect
import pybullet


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None, hostName=None, options=''):
        """Creates a Bullet client and connects to a simulation.

        Args:
          connection_mode:
            `None` connects to an existing simulation or, if fails, creates a
              new headless simulation,
            `pybullet.GUI` creates a new simulation with a GUI,
            `pybullet.DIRECT` creates a headless simulation,
            `pybullet.SHARED_MEMORY` connects to an existing simulation.
        """
        self._shapes = {}
        self._pid = os.getpid()
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY, options=options)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT

        if hostName is None:
            self._client = pybullet.connect(connection_mode, options=options)
        else:
            self._client = pybullet.connect(connection_mode, hostName=hostName, options=options)

        members = inspect.getmembers(pybullet)
        for name, attribute in members:
            if name == 'disconnect':
                continue
            if inspect.isbuiltin(attribute):
                setattr(self, name, functools.partial(attribute, physicsClientId=self._client))
            else:
                setattr(self, name, attribute)
        # print(f"self._client : {self._client}")

    def disconnect(self):
        # print("--- disconnect")
        if self._client >= 0:
            try:
                pybullet.disconnect(physicsClientId=self._client)
                self._client = -1
            except:
                pass

    # def __del__(self):
    #     print(self._client)
    #     print(self._pid)

        # """Clean up connection if not already done."""
        # if self._client>=0 and self._pid == os.getpid():
        #     pass
