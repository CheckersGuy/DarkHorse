
    #include "CmdParser.h"

//some helpler functions

bool is_double(std::string arg){
    //
    std::regex reg("[+-]?([0-9]*[.])?[0-9]+");
    return std::regex_match(arg,reg);
}


bool is_integer(std::string arg){
    //
    std::regex reg("[0-9]+");
     return std::regex_match(arg,reg);
}

bool is_string(std::string arg){
    //if it's not one of the above, we assume it's a string
    return !is_integer(arg) && !is_double(arg);
}

   

size_t CmdParser::num_options() const{
    return options.size();
}

void CmdParser::add_option(std::string arg_list)
{
    enum Type{
        IS_STRING,IS_INTEGER,IS_DOUBLE,NONE
    };

    auto get_type = [](std::string arg){
        if(is_integer(arg)){
            return IS_INTEGER;
        }
        if(is_double(arg)){
            return IS_DOUBLE;
        }
        if(is_string(arg)){
            return IS_STRING;
        }
        return NONE;
    };

 
    Type type = NONE;
    auto arguments = split_string(arg_list, std::vector<char>{' ', ','});
    auto opt_name = arguments[0];
    if(arguments.size()==1){
        //empty vector represents boolean options
        options[opt_name]= std::vector<std::string>{};
    }else if(arguments.size()>1){
        auto first = arguments[1];
        Type type = get_type(first);
        for(auto i=2;i<arguments.size();++i){
            auto c_type = get_type(arguments[i]);
            if(c_type !=type){
                type = IS_STRING;
                break;
            }
        }
       if(type ==IS_INTEGER){
            assign_values<int>(arguments);
       }else if(type == IS_DOUBLE){
            assign_values<double>(arguments);
       }else if(type == IS_STRING){
            assign_values<std::string>(arguments);
       }
    }

}

std::vector<std::string> split_string(std::string input, std::string delim)
{
    std::vector<std::string> values;
    auto pos = input.find(delim);;
    while (!input.empty() && pos !=std::string::npos)
    {
        auto word = input.substr(0,pos);
        values.emplace_back(word);
        input =input.substr(pos+delim.length());
        pos = input.find(delim);
        if(pos == std::string::npos){
            values.emplace_back(input);
        }
    }


    return values;
}
std::vector<std::string> split_string(std::string input, std::vector<char> delims)
{
    std::vector<std::string> results;
    std::string word = "";
    for (auto i = 0; i < input.size(); ++i)
    {

        auto it = std::find(delims.begin(), delims.end(), input[i]);
        if (it != delims.end())
        {
            results.emplace_back(word);
            word = "";
        }else if(i == input.size() - 1){
            word+=input[i];
            results.emplace_back(word);
            word = "";
        }
        else
        {
            word += input[i];
        }
    }

    return results;
}

     void CmdParser::parse_command_line(){
        //parsing the command line
        std::string combined="";
        for(auto i=1;i<arg_length-1;++i){
            combined+=args[i];
            combined+=" ";
        }
        combined+=args[arg_length-1];
        auto split = split_string(combined,"--");
     
        //adding the options

        for(auto word : split){
            if(word.empty())
                continue;
            add_option(word);
        }

        
     }
