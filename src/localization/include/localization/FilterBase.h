#ifndef FILTERBASE_H
#define FILTERBASE_H

#include <Eigen/Dense>


class FilterBase
{
public:
    FilterBase();
    virtual ~FilterBase();
    virtual void predict(const Eigen::VectorXd &u, double dt) = 0; //dynamic Vector of size (X x 1)
    virtual void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H) = 0;

private:

};

#endif