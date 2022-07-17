//Parsing the command line
#include <variant>
#include <vector>
#include <string>
class CmdParser{

    private:
    const int arg_length;
    const char** args;
    using CmdType = std::variant<std::string,double,std::vector<std::string>,std::vector<double>,int, std::vector<int>


    public:
     
     CmdParser(const int argl,const char** arg):arg_length(argl),args(args){

     }

     //adding options
     //what are viable options
     //using std::variant for the possible types



}