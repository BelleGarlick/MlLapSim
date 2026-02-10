from lapsim.normalisation.transform_normalisation import Transform
from utils.test_base import TestBase


"""Test vectorising a vehicle by a key or array works as expected"""


dummy_vehicle = {
    "width": -1,
    "trackFront": 0,
    "trackRear": 1,
    "wheelBaseFront": 2,
    "wheelBaseRear": 3,
    "mass": 4,
    "KDriveFront": 5,
    "KRoll": 6,
    "tyreFriction": 7,
    "maxPower": 8,
    "CoGHeight": 9,
    "FDriveMax": 10,
    "liftCoeffFront": 11,
    "liftCoeffRear": 12,
    "VMax": 13,
    "dragCoeff": 14,
    "yawInertia": 15,
    "KBrakeFront": 16
}


class TestNormalisationVehicleVectorisation(TestBase):

    def test_vectorisation_by_key(self):
        # Here we're calling the method/object directly, but normally this
        # function is called during the normalisation and transform step.
        transform = Transform(vehicle_encoding="V1")

        # Check all numbers increase in size based on dummy vehicle
        vector = transform.transform_vehicle(dummy_vehicle)
        self.assertTrue(all([vector[i] < vector[i + 1] for i in range(len(vector) - 1)]))

    def test_vectorisation_by_array(self):
        transform = Transform(vehicle_encoding=[
            "trackFront",
            "KBrakeFront"
        ])

        # Check all numbers increase in size based on dummy vehicle
        vector = transform.transform_vehicle(dummy_vehicle)
        self.assertListEqual(vector, [0, 16])

        transform = Transform(vehicle_encoding=[
            "FDriveMax",
            "CoGHeight",
            "maxPower"
        ])

        # Check all numbers increase in size based on dummy vehicle
        vector = transform.transform_vehicle(dummy_vehicle)
        self.assertListEqual(vector, [10, 9, 8])

    def test_vectorise_vehicle_list(self):
        """Test vectorising list of vehicles"""
        transform = Transform(vehicle_encoding=["test_1", "test_2"])

        vectors = transform.vectorise_vehicles([
            {"test_1": 0, "test_2": 1},
            {"test_1": 2, "test_2": 3},
            {"test_1": 4, "test_2": 5}
        ])

        self.assertListEqual(vectors[0], [0, 1])
        self.assertListEqual(vectors[1], [2, 3])
        self.assertListEqual(vectors[2], [4, 5])
