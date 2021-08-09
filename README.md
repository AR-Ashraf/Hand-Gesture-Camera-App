
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT DESCRIPTION -->
<br />

  <h3 align="center">Flutter Map</h3>

  <p align="center">
    A flutter project implementing google map API with customized marker like AIRBNB.
    <br />
    <br />
    <a href="https://github.com/AR-Ashraf/flutter_map/issues">Report Bug</a>
    ·
    <a href="https://github.com/AR-Ashraf/flutter_map/issues">Request Feature</a>
  </p>
  
  
![Screenshot 2021-07-28 174715](https://user-images.githubusercontent.com/65129467/127317430-cbea20c6-b958-422a-92c2-c6cf68d798e5.png)


  


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Flutter development is increasing day by day for its multi-platform flexibilities. In today's project I have developed a map app using google map API. The special part is that, I have customized the marker of the map like airbnb. This will help those who want to have customization in their maps.

If you are here because:
* You want to learn about implementing map feature in flutter using bloc pattern
* You don't want to roam around sites to learn map customization

Then you are at the right repo. Let's build something amazing then. If you get the time and love my repo then please consider giving it a star and follow :)


![Screenshot 2021-07-28 174742](https://user-images.githubusercontent.com/65129467/127317503-86aaddec-c65a-4612-9274-5e13af0eb3f5.png)



A list of commonly used resources that I find helpful are listed in the acknowledgements.

## Built With

* [Android-Studio](https://developer.android.com/studio)
* [Flutter](https://flutter.dev/docs/get-started/install)


<!-- GETTING STARTED -->
## Getting Started

To get started with this project to run locally follow the steps below. After that you will be ready to go.

### Prerequisites

Please clone the repo using the following command in your git bash. Or you choose to use the Clone option at the top right corner of this page to copy the link and clone it into your IDE directly.
* Clone the repo
  ```sh
  git clone https://github.com/AR-Ashraf/flutter_map.git
  ```
Open the project from your cloned directory in Android Studio. [Make sure you have installed flutter and dart]

### Installation

1. Go to pubspec.yaml file and click on pub get; or run the following commands in the terminal
   ```sh
   flutter doctor  
   ```
   ```sh
      flutter pub get 
   ```
2. Make sure your flutter sdk path and module sdk path are correctly selected in the project structure option from File tab.
  
3. Select project mode to show your code directories at the left side of android studio.
   
   Go to your android manifest file: "flutter_map/android/app/src/main/AndroidManifest.xml"
   Now add your google API key in the meta-data tag inside application tag:
   ```sh
   <meta-data android:name="com.google.android.geo.API_KEY"
              android:value="Your-API-Key"/>
   ```
   
   Again go to your ios appdelegate file: "flutter_map/ios/Runner/AppDelegate.swift"
   Now add your google API key in the bool function: 
   ```sh
      GMSServices.provideAPIKey("Your-API-Key")
   ```
   
4. Now you are ready to run the app.



<!-- USAGE EXAMPLES -->
## Usage

Use this project to make awesome map features in flutter apps.

_For more examples, please refer to the [Link](https://github.com/search?q=flutter+map)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/AR-Ashraf/flutter_map/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTION -->
## Contribution

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Dev Branch (`git checkout -b dev/flutter_map`)
3. Commit your Changes (`git commit -m 'Add some Flutter Map Feature'`)
4. Push to the Branch (`git push origin dev/flutter_map`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

LinkedIn - [Ashraful Islam](https://linkedin.com/in/ashraful-islam-78aa7a1a0) - Send me a connection request and let's get to know each other.

My Other Projects: [Projects](https://github.com/AR-Ashraf?tab=repositories)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [YouTube Link that helped me](https://www.youtube.com/watch?v=gTHKFRRSPss&t=665s)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AR-Ashraf/flutter_map.svg?style=for-the-badge
[contributors-url]: https://github.com/AR-Ashraf/flutter_map/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AR-Ashraf/flutter_map.svg?style=for-the-badge
[forks-url]: https://github.com/AR-Ashraf/flutter_map/network/members
[stars-shield]: https://img.shields.io/github/stars/AR-Ashraf/flutter_map.svg?style=for-the-badge
[stars-url]: https://github.com/AR-Ashraf/flutter_map/stargazers
[issues-shield]: https://img.shields.io/github/issues/AR-Ashraf/flutter_map.svg?style=for-the-badge
[issues-url]: https://github.com/AR-Ashraf/flutter_map/issues
[license-shield]: https://img.shields.io/github/license/AR-Ashraf/flutter_map.svg?style=for-the-badge
[license-url]: https://github.com/AR-Ashraf/flutter_map/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ashraful-islam-78aa7a1a0

