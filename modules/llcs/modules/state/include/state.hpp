#include "llcs/common.hpp"

class State {
    private:
        std::function<void()> action_function;
        std::function<int()> transition_function;
        StateType type;
        std::vector< std::tuple<int, State *> > transitions;
        

    protected:

    public:
        State( std::function<void()> ac_func, std::function<int()> tr_func, State *fault_transition );
        ~State();
        void setType(StateType type);
        std::string getType();
        void resetTransitionMap();
        void addStateTransition( int data, State *state_transition );
        void executeState();
        State* getNextState();
        State* getNextProgrammableState(int trig_num);
        void printState();
        int getTransitionSize();
};