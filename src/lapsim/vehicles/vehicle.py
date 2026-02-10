from pydantic import BaseModel, Field


"""
This module stores the vehicle class which contains the structure to store
all vehicle parameters. This class also has parameters for saving and loading 
to json/csv. If you check out the class you'll be able to see the parameters
that can be stored.

# Constructor
vehicle = Vehicle(weight=..., ..., )

# Set value based on string key - used when loading from file.
vehicle[_key_] = _val_
"""


# key_order = [
#     "trackFront",  # Track width front (m)
#     "trackRear",  # Track width rear (m)
#     "wheelBaseFront",
#     "wheelBaseRear",
#     "mass",
#     "KDriveFront",  # [-] portion of driving force at the front axle of the total driving force
#     "KRoll",  # [-] portion of roll moment at the front axle of the total roll moment
#     "tyreFriction",  # mue: constant friction coefficient (determines tire's D parameter of MF by D = F_z * mue)
#     "maxPower",
#     "CoGHeight",  # Cog Height (m)
#     "FDriveMax",  # [N] maximal drive force
#     "liftCoeffFront",  # [kg*m2/m3] lift coefficient front axle calculated by 0.5 * rho_air * c_l_f * A_spoiler_f
#     "liftCoeffRear",  # [kg*m2/m3] lift coefficient rear axle calculated by 0.5 * rho_air * c_l_r * A_spoiler_r
#     "VMax",  # maximal vehicle speed
#     "dragCoeff",  # [kg*m2/m3] drag coefficient calculated by 0.5 * rho_air * c_w * A_front
#     "yawInertia",
#     # "gamma_y",
#     # "f_z0",  # [N] nominal normal force
#     "KBrakeFront",
#     # "f_brake_max"
# ]


class Vehicle(BaseModel):

    width: float = Field(
        description="The width of the vehicle. (m)",
        default=0
    )

    track_front: float = Field(
        alias="trackFront",
        description="The front track width of the vehicle. (m)",
        default=0
    )

    track_rear: float = Field(
        alias="trackRear",
        description="The rear track width of the vehicle. (m)",
        default=0
    )

    wheel_base_front: float = Field(
        alias="wheelBaseFront",
        description="The front wheelbase of the vehicle. (m)",
        default=0
    )

    wheel_base_rear: float = Field(
        alias="wheelBaseRear",
        description="The rear wheelbase of the vehicle. (m)",
        default=0
    )

    mass: float = Field(description="The mass of the vehicle. (kg)", default=0)

    # portion of driving force at the front axle of the total driving force
    k_drive_front: float = Field(
        alias="KDriveFront",
        default=0,
        description="Front suspension's drive stiffness.")

    k_roll: float = Field(
        alias="KRoll",
        default=0,
        description="Roll stiffness of the vehicle's suspension system.")

    # mue: constant friction coefficient (determines tire's D parameter of MF by D = F_z * mue)
    tyre_friction: float = Field(
        alias="tyreFriction",
        default=0,
        description="Coefficient of friction for the vehicle's tires.")

    max_power: float = Field(
        alias="maxPower",
        default=0,
        description="Maximum engine power of the vehicle.")

    cog_height: float = Field(
        alias="CoGHeight",
        description="Height of the vehicle's center of gravity. (m)",
        default=0)

    f_drive_max: float = Field(
        alias="FDriveMax",
        description="Maximum drive force that can be applied. (N)",
        default=0)

    # [kg*m2/m3] lift coefficient front axle calculated by 0.5 * rho_air * c_l_f * A_spoiler_f
    lift_coeff_front: float = Field(
        alias="liftCoeffFront",
        default=0,
        description="Coefficient of lift for the front of the vehicle.")

    # [kg*m2/m3] lift coefficient rear axle calculated by 0.5 * rho_air * c_l_r * A_spoiler_r
    lift_coeff_rear: float = Field(
        alias="liftCoeffRear",
        default=0,
        description="Coefficient of lift for the rear of the vehicle.")

    v_max: float = Field(
        alias="VMax",
        description="Maximum velocity or top speed of the vehicle.",
        default=0)

    # [kg*m2/m3] drag coefficient calculated by 0.5 * rho_air * c_w * A_front
    drag_coeff: float = Field(
        alias="dragCoeff",
        default=0,
        description="Coefficient of drag for aerodynamic resistance.")

    yaw_inertia: float = Field(
        alias="yawInertia",
        default=0,
        description="Yaw inertia or rotational inertia of the vehicle.")

    k_brake_front: float = Field(
        alias="KBrakeFront",
        default=0,
        description="Front brake system's stiffness.")

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise Exception(f"Unknown key: '{key}'")
        setattr(self, key, value)

    def to_array(self):
        return [
            self.width,
            self.track_front,
            self.track_rear,
            self.wheel_base_front,
            self.wheel_base_rear,
            self.mass,
            self.k_drive_front,
            self.k_roll,
            self.tyre_friction,
            self.max_power,
            self.cog_height,
            self.f_drive_max,
            self.lift_coeff_front,
            self.lift_coeff_rear,
            self.v_max,
            self.drag_coeff,
            self.yaw_inertia,
            self.k_brake_front
        ]
