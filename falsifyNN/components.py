# This file defines class objects for Roads, Sceneries and Cars.

from collections import namedtuple

RoadDefn = namedtuple("road", "road_imageFile description vp_x vp_y")
CarDefn = namedtuple("car", "car_imageFile description")

class road(RoadDefn):
	def __repr__(self):
		return "Picture : " + self.description + "\nVanishing Point : " + str([self.vp_x, self.vp_y])


class car(CarDefn):
	def __repr__(self):
		return self.description
