#ifndef CONTROLLER_
#define CONTROLLER_

class Controller {

public:

    virtual ~Controller() = default;

    virtual void reset() = 0;

    virtual double compute_correction(double v) = 0;
};

#endif