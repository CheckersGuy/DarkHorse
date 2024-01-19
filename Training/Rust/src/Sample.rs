//move sample definition from dataloader here
//should make stuff a little easier to handle :)
use crate::Pos::Position;
use crate::Pos::Square;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use byteorder::WriteBytesExt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
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
pub struct Sample {
    pub position: SampleType,
    pub eval: i16,
    pub result: Result,
}

impl Sample {
    pub fn write_fen<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        if let SampleType::Fen(ref fen_string) = self.position {
            let length: u16 = fen_string.len() as u16;
            writer.write_u16::<LittleEndian>(length)?;
            writer.write_all(fen_string.as_bytes())?;
            writer.write_i16::<LittleEndian>(self.eval)?;
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
        }

        Ok(())
    }

    pub fn read_into<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // to be added
        let length: u16 = reader.read_u16::<LittleEndian>()?;
        let mut buffer = vec![0; length as usize];
        reader.read_exact(&mut buffer)?;
        self.position = SampleType::Fen(String::from_utf8(buffer).unwrap());
        self.eval = reader.read_i16::<LittleEndian>()?;
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
//refactoring idea: GameIterator should have a member next_game and return & instead
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

            let squares = sample.position.get_squares().unwrap();
            let piece_count = squares.len();
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
        let _num_samples = self
            .read_u64::<LittleEndian>()
            .expect("Could not read number of samples");
        SampleIterator { reader: self }
    }

    fn iter_games(&'a mut self) -> GameIterator<'a> {
        let _num_samples = self
            .read_u64::<LittleEndian>()
            .expect("Could not read number of samples");
        GameIterator {
            reader: self,
            game: Vec::new(),
        }
    }
}

impl From<(Position, Result)> for Sample {
    fn from(value: (Position, Result)) -> Self {
        Sample {
            result: value.1,
            position: SampleType::Pos(value.0),
            ..Sample::default()
        }
    }
}
