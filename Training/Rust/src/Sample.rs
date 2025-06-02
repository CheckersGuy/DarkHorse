//move sample definition from dataloader here
//should make stuff a little easier to handle :)
use crate::Pos::MoveList;
use crate::Pos::Position;
use crate::Pos::Square;
use bloomfilter::reexports::bit_vec::BitBlock;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::str::FromStr;
#[derive(Debug, Clone, Hash, PartialEq)]
pub enum SampleType {
    Fen(String), //a not yet converted FenString
    Pos(Position),
    Squares(Vec<Square>),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Result {
    UNKNOWN,
    WIN,
    LOSS,
    DRAW,
    TBWIN,
    TBLOSS,
    TBDRAW,
}

impl Default for Result {
    fn default() -> Self {
        Result::UNKNOWN
    }
}

impl Default for SampleType {
    fn default() -> Self {
        SampleType::None
    }
}

impl From<i8> for Result {
    fn from(item: i8) -> Self {
        match item {
            1 => Result::LOSS,
            2 => Result::WIN,
            3 => Result::DRAW,
            4 => Result::TBLOSS,
            5 => Result::TBWIN,
            6 => Result::TBDRAW,
            _ => Result::UNKNOWN,
        }
    }
}

pub struct Game {
    start_pos: Position,
    moves: Vec<u8>,
    result: Result,
    current: Position,
}

impl Game {
    pub fn new() -> Game {
        Game {
            start_pos: Position::empty(),
            moves: Vec::new(),
            result: Result::UNKNOWN,
            current: Position::empty(),
        }
    }

    pub fn set_start_position(&mut self, pos: Position) {
        self.start_pos = pos;
        self.current = pos;
    }
    //should return an error if the position could not be found
    pub fn add_position(&mut self, pos: Position) -> Option<u8> {
        let mut liste = MoveList::new();
        liste.get_moves(self.current);
        for (index, m) in liste.iter().enumerate() {
            let mut copy = self.current;
            copy.make_move(m);
            if copy == pos {
                return Some(index as u8);
            }
        }
        None
    }

    pub fn save_game<W: Write>(&self, writer: &mut W) {
        //writing the length of the game first
        writer
            .write_u16::<LittleEndian>(self.moves.len() as u16)
            .expect("Could not write game_length");
        writer
            .write_u32::<LittleEndian>(self.start_pos.wp)
            .expect("Could not write wp");
        writer
            .write_u32::<LittleEndian>(self.start_pos.bp)
            .expect("Could not write bp");
        writer
            .write_u32::<LittleEndian>(self.start_pos.k)
            .expect("Could not write k");
        writer
            .write_i8(self.start_pos.color)
            .expect("Could not write color");
        let conv = match self.result {
            Result::LOSS => 1,
            Result::WIN => 2,
            Result::DRAW => 3,
            Result::TBLOSS => 4,
            Result::TBWIN => 5,
            Result::TBDRAW => 6,
            Result::UNKNOWN => 0,
        };
        writer.write_i8(conv).expect("Could not write color");

        for ind in self.moves.iter() {
            writer.write_u8(*ind).expect("Could not write ind");
        }
    }

    pub fn read_game<R: Read>(&mut self, reader: &mut R) {
        //the length of the game does not include any captures or the starting position of the game
        self.moves.clear();
        let game_length = reader
            .read_u16::<LittleEndian>()
            .expect("Could not read game_length");
        let wp = reader
            .read_u32::<LittleEndian>()
            .expect("Could not read wp");
        let bp = reader
            .read_u32::<LittleEndian>()
            .expect("Could not read bp");
        let k = reader.read_u32::<LittleEndian>().expect("Could not read k");
        let color = reader.read_i8().expect("Could not read color");
        //reading the game_result;

        let result_index = reader.read_i8().expect("Could not read result_index");
        self.result = match result_index {
            1 => Result::LOSS,
            2 => Result::WIN,
            3 => Result::DRAW,
            4 => Result::TBLOSS,
            5 => Result::TBWIN,
            6 => Result::TBDRAW,
            _ => Result::UNKNOWN,
        };

        let mut pos = Position::empty();

        pos.wp = wp;
        pos.bp = bp;
        pos.k = k;
        pos.color = color;
        self.start_pos = pos;

        for _ in 0..game_length {
            let move_index = reader.read_u8().expect("Could not read move_index");
            self.moves.push(move_index);
        }
    }
}

impl std::ops::Not for Result {
    type Output = Self;
    fn not(self) -> Self::Output {
        match self {
            Result::WIN => Result::LOSS,
            Result::LOSS => Result::WIN,
            Result::DRAW => Result::DRAW,
            Result::TBLOSS => Result::TBWIN,
            Result::TBWIN => Result::TBLOSS,
            Result::TBDRAW => Result::TBDRAW,
            Result::UNKNOWN => Result::UNKNOWN,
            _ => self,
        }
    }
}
//below needs to be tested
impl FromStr for Result {
    type Err = anyhow::Error;
    fn from_str(item: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        match item
            .to_lowercase()
            .trim()
            .replace("\n", "")
            .replace("_", "")
            .as_str()
        {
            "loss" | "lost" => Ok(Result::LOSS),
            "tbloss" | "tblost" => Ok(Result::TBLOSS),
            "tbwin" | "tbwon" => Ok(Result::TBWIN),
            "tbdraw" | "tbdrew" => Ok(Result::TBDRAW),
            "win" | "won" => Ok(Result::WIN),
            "draw" | "drew" => Ok(Result::DRAW),
            _ => Err(anyhow::anyhow!("Could not parse sample")),
        }
    }
}

impl From<&str> for Result {
    fn from(item: &str) -> Self {
        match item {
            "loss" | "LOSS" | "LOST" | "lost" => Result::LOSS,
            "TB_LOSS" | "TB_LOST" | "TBLOSS" | "TBLOST" => Result::TBLOSS,
            "TB_WIN" | "TB_WON" | "TBWIN" | "TBWON" => Result::TBWIN,
            "TBDRAW" | "TB_DREW" | "TBDREW" => Result::TBDRAW,
            "win" | "WIN" | "WON" | "won" => Result::WIN,
            "DRAW" | "draw" | "TB_DRAW" => Result::DRAW,
            _ => Result::UNKNOWN,
        }
    }
}

impl SampleType {
    pub fn get_squares(&self) -> std::io::Result<Vec<Square>> {
        let mut squares = Vec::new();
        let fen_string = match self {
            SampleType::Fen(ref fen) => fen,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Error parsing color",
                ))
            }
        };

        let _color = match fen_string.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Error parsing color",
                ))
            }
        };

        //to be continued
        //need to convert the option in next.unwrap() to a Result
        for s in fen_string.split(":").skip(1) {
            let mut color: i32 = 0;
            let token_op = s.chars().next();
            if token_op == None {
                std::io::Error::new(std::io::ErrorKind::NotFound, "Error parsing color");
            }
            match token_op {
                Some('W') => color = 1,
                Some('B') => color = -1,
                _ => (),
            }
            let splits = s.split(",");

            for (i, val) in splits.enumerate() {
                let mut sq_str = val.chars();
                if i == 0 {
                    sq_str.next();
                }
                let m = sq_str.clone().next().unwrap();
                match m {
                    'K' => {
                        sq_str.next();
                        let square: u8 = match sq_str.as_str().parse() {
                            Ok(n) => n,
                            Err(_) => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        };
                        squares.push(match color {
                            1 => Square::WKING(square - 1),
                            -1 => Square::BKING(square - 1),
                            _ => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        });
                    }

                    _ => {
                        let square: u8 = match sq_str.as_str().parse() {
                            Ok(n) => n,
                            Err(_) => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        };
                        squares.push(match color {
                            1 => Square::WPAWN(square - 1),
                            -1 => Square::BPAWN(square - 1),
                            _ => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        });
                    }
                }
            }
        }
        Ok(squares)
    }
}

#[derive(Default, Clone, Hash, PartialEq, Debug)]
//there always should be a mlh-value for every sample <- sounds like I need to add an assert
//somewhere

pub struct Sample {
    pub position: Position,
    pub mlh: i16,
    pub result: Result,
}

impl Sample {
    pub fn write_fen<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.position.wp)?;
        writer.write_u32::<LittleEndian>(self.position.bp)?;
        writer.write_u32::<LittleEndian>(self.position.k)?;
        writer.write_i8(self.position.color)?;
        writer.write_i16::<LittleEndian>(self.mlh)?;
        let conv = match self.result {
            Result::LOSS => 1,
            Result::WIN => 2,
            Result::DRAW => 3,
            Result::TBLOSS => 4,
            Result::TBWIN => 5,
            Result::TBDRAW => 6,
            Result::UNKNOWN => 0,
        };
        writer.write_i8(conv)?;

        Ok(())
    }

    pub fn read_into<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // to be added
        self.position.wp = reader.read_u32::<LittleEndian>()?;
        self.position.bp = reader.read_u32::<LittleEndian>()?;
        self.position.k = reader.read_u32::<LittleEndian>()?;
        self.position.color = reader.read_i8()?;
        self.mlh = reader.read_i16::<LittleEndian>()?;
        let conv = reader.read_i8()?;
        self.result = match conv {
            1 => Result::LOSS,
            2 => Result::WIN,
            3 => Result::DRAW,
            4 => Result::TBLOSS,
            5 => Result::TBWIN,
            6 => Result::TBDRAW,
            _ => Result::UNKNOWN,
        };

        Ok(())
    }
}

pub struct SampleIterator<'a> {
    reader: &'a mut BufReader<File>,
}

pub struct GameIterator<'a> {
    reader: &'a mut BufReader<File>,
    game: Vec<Sample>,
}

//iterator needs to be tested
pub trait SampleIteratorTrait<'a> {
    //fn iterate_samples();
    fn iter_samples(&'a mut self) -> SampleIterator<'a>;
    fn iter_games(&'a mut self) -> GameIterator<'a>;
}

impl<'a> Iterator for SampleIterator<'a> {
    type Item = Sample;
    fn next(&mut self) -> Option<Self::Item> {
        let mut sample = Sample::default();
        let result = sample.read_into(&mut self.reader);
        match result {
            Ok(_) => Some(sample),
            Err(_) => None,
        }
    }
}
impl<'a> Iterator for GameIterator<'a> {
    type Item = Vec<Sample>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut prev_count = 0;
        loop {
            let mut sample = Sample::default();

            match sample.read_into(&mut self.reader) {
                Ok(_) => {}
                Err(_) => break,
            };

            let piece_count = sample.position.bp.count_ones() + sample.position.wp.count_ones();
            if piece_count >= prev_count {
                self.game.push(sample);
            } else {
                let ret_val = Some(self.game.clone());
                self.game.clear();
                self.game.push(sample);
                return ret_val;
            }
            prev_count = piece_count;
        }
        return None;
    }
}
//to be implemented
impl<'a> SampleIterator<'a> {
    fn consume<W: Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        while let Some(sample) = self.next() {
            sample.write_fen(writer)?;
        }
        Ok(())
    }
}

impl<'a> SampleIteratorTrait<'a> for BufReader<File> {
    fn iter_samples(&'a mut self) -> SampleIterator<'a> {
        SampleIterator { reader: self }
    }

    fn iter_games(&'a mut self) -> GameIterator<'a> {
        GameIterator {
            reader: self,
            game: Vec::<Sample>::new(),
        }
    }
}
